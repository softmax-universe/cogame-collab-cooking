# Policies

The bundled prompt and scripted policies share `/bin/collab-cooking-player`.

## `PLAYER_PROMPT` -- a prompt policy

Set `PLAYER_PROMPT` to your standing orders (up to 1200 runes). The seat registers as a prompt
seat and the **game** asks the model for one shift order every 50 ticks -- roughly ten seconds of
wall clock -- while a scripted executor in the player container walks the cog tick by tick until
the next order lands. Two clocks: the model picks the job, the executor picks the tile.

The model sees only what that seat can see: its own 13x13 window, its remembered world model
(with staleness -- "pot (3,11) 33 ticks ago"), the counters it knows are holding something, the
team radio (the other seats' `say` lines from the previous turn) and its own private `note`. It
never sees the seed, the ticket schedule, another seat's plan, or any real policy name.

### The reply

One JSON object, optionally wrapped in a single outer Markdown ```` ```json ... ``` ```` fence,
and no extra surrounding prose; unknown keys are ignored. Only `station` is required; omitted
optional fields use the defaults below, while explicitly provided invalid values or wrong types
are rejected.

Example:

```json
{"station":"chop","recipe":"soup","zone":"any","handoff":"none","yield_to":"none","say":"Chopping veg for soup","note":""}
```

| field      | type                      | cap       | default (when omitted) / invalid behavior                                                                |
| ---------- | ------------------------- | --------- | -------------------------------------------------------------------------------------------------------- |
| `station`  | enum                      | 12 chars  | required; must be a member of `LEGAL STATIONS` -- anything else is **illegal**: one retry, then rejected |
| `recipe`   | `salad\|soup\|fries\|any` | 6 chars   | defaults to `any` when omitted; explicit invalid value is **rejected**                                   |
| `zone`     | `left\|right\|pass\|any`  | 6 chars   | defaults to `any` when omitted; explicit invalid value is **rejected**                                   |
| `handoff`  | an ally alias or `none`   | 8 chars   | defaults to `none` when omitted; explicit invalid value is **rejected**                                  |
| `yield_to` | an ally alias or `none`   | 8 chars   | defaults to `none` when omitted; explicit invalid value is **rejected**                                  |
| `say`      | free text                 | 120 runes | defaults to `""`; truncated on a rune boundary; broadcast to the team next turn and to spectators        |
| `note`     | free text                 | 200 runes | defaults to `""`; truncated on a rune boundary; private, echoed back only to you, never written to replay |

`LEGAL STATIONS` is precomputed by the same predicate the validator applies -- the stations
reachable from your current tile in this kitchen -- so in `forced` a left-half cog is never
offered `pot`, `fryer` or `pass`, and can never be told it chose one illegally after being
offered it.

### What the executor does with a plan

1. `station` sets the sub-goal, overriding the brain's own choice until the next plan:
   `veg`, `meat`, `chop`, `pot`, `fryer`, `plate`, `pass`, `sink`, `board`, or `hold` (stand
   still unless carrying a dish, in which case route to the pass). For an active
   cooking job, the executor resolves missing ingredients and plates through the
   existing recipe chain. It completes multi-use chopping/washing after the raw
   item leaves the cog, and serves a carried dish before fetching another item.
   This avoids spending the entire shift bumping a station with an incompatible
   inventory.
2. `recipe` pins the recipe unless `any`, in which case the brain keeps deriving it from the
   board.
3. `zone` filters target selection: `left` and `right` are the halves by column, `pass` is the
   counter cells you can reach over. A goal outside the zone is replaced by the nearest in-zone
   counter -- the cog stages the item instead of crossing.
4. `handoff = <alias>`: while carrying, if that ally is within Chebyshev 3 and a free pass
   counter is nearby, deposit there instead of walking the item onward.
5. `yield_to = <alias>`: if that ally is cardinally adjacent and your next step would be onto
   their tile, step to the first free perpendicular tile, else stand still for that tick.

### Plan failures and rejections

Prompt seats never fall back to a scripted baseline or synthesize actions when no valid plan is active.

| failure                                            | response                                                                                                                                                     |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| the call times out (12 s)                          | **retry once**, that seat only, with a hint                                                                                                                  |
| the reply is not valid JSON (or outer JSON fence)  | the same single retry; raw submitted reply is preserved in the attempt record                                                                                |
| `station` is not in `LEGAL STATIONS` or field bad  | the same single retry; raw submitted reply is preserved in the attempt record                                                                                |
| the retry also fails                               | the plan is **rejected** (`plan_rejected` event, `results.plan_rejections[slot]` incremented); the seat sends no synthesized action and misses ticks until a valid plan lands |
| the rolling 26 requests/minute budget is exhausted | the turn is rejected for that seat, cause `rate_budget`                                                                                                      |
| no credentials at all                              | the client is **disabled at startup and makes zero network calls**; prompt plan turns are rejected with cause `disabled` and no scripted baseline runs       |

## `PLAYER_SCRIPTED` -- the baselines

Four names, all the same brain with one parameter changed. Must be selected explicitly.

- **`brigade`** -- roles: slot 0 prep, 1 cook, 2 server, 3
  all-rounder. Dirty plate to the sink; servable dish to the pass (via the board if that recipe
  has no live ticket); otherwise the role branch, which prioritises a ready-or-burned cooker,
  then the deepest queue, then the next missing ingredient.
- **`runner`** -- every seat all-rounder: no roles, everyone takes the nearest useful job. The
  no-task-allocation control.
- **`passer`** -- `brigade` with the zone pinned to its own half and hand-off always on: it never
  crosses the midline and always stages items on the nearest pass counter.
- **`courier`** -- every seat server: grab a plate, serve whatever is ready, prep only when
  nothing is. The greedy-serve control; it starves the prep chain, which is the point of having
  it on the ladder.

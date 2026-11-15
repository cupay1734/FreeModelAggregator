# FreeModelAggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/cupay1734/FreeModelAggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/cupay1734/FreeModelAggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves an OpenAI- and an Anthropic-compatible API over ten free-tier
model providers. It reads the providers' catalogs, routes every request to the
highest-rated free model that meets the request's requirements (tools, vision, JSON
mode, context length) and still has budget, speaks that provider's own wire (OpenAI,
Anthropic or Gemini), fails over to the next candidate when a provider refuses, and
returns the answer in the wire the client used, streamed or not. Routing is a pure
function of the requests and the answers already seen: no wall clock, no randomness,
so a replay walks the same routes.

## Run it

```sh
node main.ts          # serves; needs MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE
node main.ts --list   # prints the catalog the server registers, and exits
```

Node 22, standard library only.

---

<sub>Entered on [make.fun](https://make.fun/?ref=s522t60h&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>

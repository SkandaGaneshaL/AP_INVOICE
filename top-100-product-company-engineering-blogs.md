# Top 100 Product-Company Engineering Blogs

Direct sources for **production-grade architecture, system design, scalable systems, and LLM/agent application logic**.

Compiled: **2 September 2026**. Product companies only (not consultancies, not courses). Each URL is the engineering-blog home — open it and search `agent`, `LLM`, `RAG`, `eval`, `serving`, `ranking`, `feature store`.

**How to use this as one bench**

- Read the **active 2025–2026** blogs first (Meta, Google Cloud, AWS, Uber, Stripe, Shopify, Netflix, Databricks, Anthropic, OpenAI, Hugging Face, DoorDash, Figma, Vercel, Nubank).
- Treat **X / Twitter engineering** as historical — last real posts ~2023.
- Pair a company blog with the original papers from the previous source list when they name an algorithm (HNSW, LoRA, ReAct, GraphRAG, PagedAttention).

---

## The table (100)

| # | Company | Engineering blog | What they publish (AI / systems / scale) |
| - | ------- | ---------------- | ---------------------------------------- |
| 1 | Meta | [engineering.fb.com](https://engineering.fb.com/) | AI infra, PyTorch/FAISS, ranking, capacity agents, tribal-knowledge agents, LLM inference |
| 2 | Meta AI | [ai.meta.com/blog](https://ai.meta.com/blog/) | Llama, self-supervised learning, recommendation, multimodal research shipped into products |
| 3 | Google Cloud | [cloud.google.com/blog](https://cloud.google.com/blog) | Gemini Enterprise Agent Platform, Agent Executor, attested computations, MCP, Vertex/Agent Runtime |
| 4 | Google Research | [research.google/blog](https://research.google/blog/) | Gemini, Pathways, retrieval, long-context, TPU serving, research that lands in Google products |
| 5 | Google DeepMind | [deepmind.google/discover/blog](https://deepmind.google/discover/blog/) | Frontier model research, Alpha-series, agents, science, Gemini internals |
| 6 | AWS Architecture | [aws.amazon.com/blogs/architecture](https://aws.amazon.com/blogs/architecture/) | AgentCore, graduated autonomy, multi-agent clusters, lakehouse-for-agents reference architectures |
| 7 | AWS Machine Learning | [aws.amazon.com/blogs/machine-learning](https://aws.amazon.com/blogs/machine-learning/) | Bedrock agents, RAG on OpenSearch, SageMaker serving, AWS Context / knowledge graphs for agents |
| 8 | Amazon Science | [amazon.science/blog](https://www.amazon.science/blog) | Alexa, retrieval, GNN training, AutoML, search ranking, production research |
| 9 | Netflix | [netflixtechblog.com](https://netflixtechblog.com/) | Agentic causal inference, data canaries, live ops, recsys, ML workflow orchestration |
| 10 | Uber | [eng.uber.com](https://eng.uber.com/) | Agentic SDLC, secrets/identity for agents, MCP gateway (1000+ servers), AI Context Graph, cost-per-token |
| 11 | LinkedIn | [linkedin.com/blog/engineering](https://www.linkedin.com/blog/engineering) | QA Agent, AI code review, search stack, Graph Neural Nets, feed ranking, MLOps portal |
| 12 | Stripe | [stripe.dev/blog](https://stripe.dev/blog) | Minions coding agents, knowledge-AI platform, ledger, auto-remediation, fraud ML, distributed proxy |
| 13 | X / Twitter | [blog.x.com/engineering](https://blog.x.com/engineering/en_us) | **Stale — last real eng posts ~2023.** Historical: recsys, GNN, Hadoop, users DB scale |
| 14 | Discord | [discord.com/blog](https://discord.com/blog) | Cost attribution, voice outage, tracing, Elixir/Rust infra, ML moderation |
| 15 | Dropbox | [dropbox.tech](https://dropbox.tech/) | Nova coding-agent platform, cookie auditor, MCP, OCR, content search, ML for files |
| 16 | Databricks | [databricks.com/blog](https://www.databricks.com/blog) | AI incident investigation, Mosaic/DVR, agent eval (MLflow traces), agentic Postgres, Unity Catalog for agents |
| 17 | GitHub | [github.blog/engineering](https://github.blog/engineering/) | Copilot agents, stacked PRs, review cost, code-search, Actions at scale |
| 18 | Shopify | [shopify.engineering](https://shopify.engineering/) | River agent, gisting (long-prompt compression), Flow-from-language, computer-vision E2E tests |
| 19 | Airbnb | [medium.com/airbnb-engineering](https://medium.com/airbnb-engineering) | LLM eval speed, auth, forecast unlearning, search ranking, conversational AI |
| 20 | Canva | [canva.dev/blog/engineering](https://www.canva.dev/blog/engineering/) | Session revocation at scale, design-AI pipelines, multi-region identity |
| 21 | Lyft | [eng.lyft.com](https://eng.lyft.com/) | Localization pipeline, feature store, metrics layer, causal forecasting, pricing ML |
| 22 | Pinterest | [medium.com/pinterest-engineering](https://medium.com/pinterest-engineering) | Retrieval, resource provisioner, foundation models, ads ranking, visual search |
| 23 | Zomato | [blog.zomato.com](https://blog.zomato.com/) | Gavel eval infra, real-time ads on Flink, hyperlocal search, food-delivery systems |
| 24 | Spotify | [engineering.atspotify.com](https://engineering.atspotify.com/) | Background coding agents, context layer, agentic DX, personalization, experimentation at scale |
| 25 | Hotstar / JioHotstar | [blog.hotstar.com](https://blog.hotstar.com/) | 24/7 live-sports watchdog, CDN traffic shaping, 60M concurrent-user infra |
| 26 | Apple Machine Learning | [machinelearning.apple.com](https://machinelearning.apple.com/) | On-device LLMs, Apple Intelligence, Core ML, private compute, Foundation Models |
| 27 | Microsoft Engineering | [devblogs.microsoft.com/engineering-at-microsoft](https://devblogs.microsoft.com/engineering-at-microsoft/) | Copilot in Microsoft 365, GitHub Copilot internals, Azure-scale identity, Windows AI |
| 28 | Microsoft Azure | [azure.microsoft.com/blog](https://azure.microsoft.com/blog/) | Azure AI Foundry, Agent Service, OpenAI on Azure, confidential inference |
| 29 | NVIDIA | [developer.nvidia.com/blog](https://developer.nvidia.com/blog/) | TensorRT-LLM, NIM, CUDA graphs, inference kernels, RAG blueprints, NeMo agents |
| 30 | OpenAI | [openai.com/news](https://openai.com/news) | GPT/Codex, Agents SDK, Structured Outputs, function calling, evals, production safety |
| 31 | Anthropic | [anthropic.com/engineering](https://www.anthropic.com/engineering) | Building effective agents, context engineering, tool design, Agent Skills, containment, evals |
| 32 | Hugging Face | [huggingface.co/blog](https://huggingface.co/blog) | Transformers, PEFT/LoRA, TRL (SFT/DPO/GRPO), TGI/TEI serving, smolagents, papers |
| 33 | Slack | [slack.engineering](https://slack.engineering/) | Search, spam/recsys, Flannel real-time, AI recap, large-scale messaging systems |
| 34 | Cloudflare | [blog.cloudflare.com](https://blog.cloudflare.com/) | Workers AI, inference at the edge, DDoS, R2, global Anycast, AI Gateway, MCP |
| 35 | Figma | [figma.com/blog/engineering](https://www.figma.com/blog/engineering/) | Multiplayer CRDTs, agentic SecOps, AI design features, wasm/GPU rendering |
| 36 | PayPal | [medium.com/paypal-tech](https://medium.com/paypal-tech) | Graph ML, fraud, explainable AI, NLP, payments at global scale |
| 37 | DoorDash | [careersatdoordash.com/engineering-blog](https://careersatdoordash.com/engineering-blog/) | Platform for building/evolving agents, ML monitoring, demand forecasting, logistics |
| 38 | Snowflake | [snowflake.com/en/blog](https://www.snowflake.com/en/blog/) | Cortex agents, governed LLM access, Snowflake Intelligence, data-for-agents |
| 39 | Salesforce | [engineering.salesforce.com](https://engineering.salesforce.com/) | Einstein / Agentforce, CRM agents, hyperforce, multi-tenant scale |
| 40 | eBay | [innovation.ebayinc.com](https://innovation.ebayinc.com/) | Search/ranking, listing understanding, marketplace ML, large catalog retrieval |
| 41 | MongoDB | [mongodb.com/blog/channel/engineering-blog](https://www.mongodb.com/blog/channel/engineering-blog) | Atlas Vector Search, Voyage embeddings, distributed storage, queryable encryption |
| 42 | Instacart | [tech.instacart.com](https://tech.instacart.com/) | Real-time ML, marketplace balancer, search embeddings, demand prediction |
| 43 | Notion | [notion.so/blog/topic/engineering](https://www.notion.so/blog/topic/engineering) | Notion AI / agents, block CRDT, search, workspace-scale sync |
| 44 | Etsy | [etsy.com/codeascraft](https://www.etsy.com/codeascraft) | Search ranking, ML platform, observability, personalized recommendations |
| 45 | Grab | [engineering.grab.com](https://engineering.grab.com/) | Data mesh, SuperApp payments, maps/routing, ML platform, SEA-scale logistics |
| 46 | Atlassian | [atlassian.com/blog/atlassian-engineering](https://www.atlassian.com/blog/atlassian-engineering) | Rovo agents, Jira/Confluence AI, multi-product identity, tenant isolation |
| 47 | Coinbase | [coinbase.com/blog/landing/engineering](https://www.coinbase.com/blog/landing/engineering) | CEEcil support agent with memory, crypto infra, custody, agentic loops |
| 48 | Flipkart | [blog.flipkart.tech](https://blog.flipkart.tech/) | MySQL HA, multi-region ZooKeeper, search, India-scale marketplace |
| 49 | Razorpay | [engineering.razorpay.com](https://engineering.razorpay.com/) | Payments auth revamp, real-time money movement, idempotency, risk ML |
| 50 | Reddit | [reddit.com/r/RedditEng](https://www.reddit.com/r/RedditEng) | Feed ranking, comment systems, ads ML, safety models, traffic spikes |
| 51 | Snap | [eng.snap.com](https://eng.snap.com/) | On-device ML, AR, ads ranking, feature stores, camera pipelines |
| 52 | Swiggy | [bytes.swiggy.com](https://bytes.swiggy.com/) | Hyperlocal search, routing, fraud rings, contextual bandits, delivery fleet |
| 53 | Walmart Global Tech | [medium.com/walmartglobaltech](https://medium.com/walmartglobaltech) | ML platform, AutoML, routing, recsys, store/e-comm omnichannel |
| 54 | Booking.com | [booking.ai](https://booking.ai/) | Ranking, causal inference, experimentation, recsys, travel ML pipelines |
| 55 | Grammarly | [grammarly.com/blog/engineering](https://www.grammarly.com/blog/engineering/) | GEC, LLM rewriting, latency-sensitive NLP, on-device + cloud hybrid |
| 56 | Datadog | [datadoghq.com/blog/engineering](https://www.datadoghq.com/blog/engineering/) | Bits AI, observability agents, APM, CI git serving at 20×, profiling |
| 57 | Vercel | [vercel.com/blog/category/engineering](https://vercel.com/blog/category/engineering) | AI SDK, Fluid compute, agent credentials, v0, edge inference |
| 58 | Nubank | [building.nubank.com/engineering](https://building.nubank.com/engineering/) | Building AI agents for 130M customers, foundation models, Clojure+AI platform |
| 59 | Klarna | [engineering.klarna.com](https://engineering.klarna.com/) | Customer-service agents, payments, LLM ops replacing large support orgs |
| 60 | Mercado Libre | [medium.com/mercadolibre-tech](https://medium.com/mercadolibre-tech) | ML platform, forecasting, causal inference, LatAm marketplace + fintech |
| 61 | Intuit | [medium.com/intuit-engineering](https://medium.com/intuit-engineering) | GenOS, tax/finance agents, TurboTax AI, document understanding |
| 62 | Yelp | [engineeringblog.yelp.com](https://engineeringblog.yelp.com/) | Search, reviews NLP, ads, photo ML, local graph |
| 63 | Twitch | [blog.twitch.tv](https://blog.twitch.tv/) | Live video, chat fan-out, recsys, safety ML, IVS-scale streaming |
| 64 | Docker | [docker.com/blog](https://www.docker.com/blog/) | BuildKit, container isolation for agents, Compose, Desktop, supply chain |
| 65 | Postman | [blog.postman.com](https://blog.postman.com/) | API agents, collections at scale, MCP, API governance |
| 66 | GitLab | [about.gitlab.com/blog](https://about.gitlab.com/blog/) | Duo agents, CI at GitLab.com scale, Elasticsearch code search |
| 67 | Elastic | [elastic.co/blog](https://www.elastic.co/blog/) | ESRE / hybrid search, ELSER sparse retrieval, observability, vector search |
| 68 | Cockroach Labs | [cockroachlabs.com/blog](https://www.cockroachlabs.com/blog/) | Distributed SQL, multi-region consensus, correctness at scale |
| 69 | Sentry | [blog.sentry.io](https://blog.sentry.io/) | Seer AI debugging, tracing, error grouping, ingest pipelines |
| 70 | Block (Square) | [engineering.block.xyz/blog](https://engineering.block.xyz/blog) | Autohealing Moneybot (LLM repair loops), Square payments, Bitcoin infra |
| 71 | Duolingo | [blog.duolingo.com/hub/engineering](https://blog.duolingo.com/hub/engineering/) | Birdbrain recsys, LLM tutors, A/B at massive DAU, production AI agents |
| 72 | NAVER | [d2.naver.com](https://d2.naver.com/) | Search, HyperCLOVA, DEVIEW talks, Korea-scale retrieval and ads |
| 73 | Alibaba Cloud | [alibabacloud.com/blog](https://www.alibabacloud.com/blog) | Qwen, PAI, ecommerce ranking, Amap, cloud-native AI engineering |
| 74 | TikTok / ByteDance | [developers.tiktok.com/blogs](https://developers.tiktok.com/blogs) | Recommendation, live, creator tools; ByteDance infra (Monolith, etc.) |
| 75 | Mercari | [engineering.mercari.com/en/blog](https://engineering.mercari.com/en/blog/) | C2C marketplace, payments (Merpay), listing ML, JP-scale microservices |
| 76 | Gojek | [gojek.io/blog](https://www.gojek.io/blog) | SuperApp, GoPay, ML platform, demand forecasting, SEA transport |
| 77 | Freshworks | [medium.com/freshworks-engineering-blog](https://medium.com/freshworks-engineering-blog) | Freddy AI, multi-tenant SaaS, Cassandra→Scylla, support agents |
| 78 | Zerodha | [zerodha.tech](https://zerodha.tech/) | Brokerage at India scale with a tiny team, Kite, Rust/Go, ops discipline |
| 79 | Mistral AI | [mistral.ai/news](https://mistral.ai/news) | Open-weight + commercial LLMs, MoE, serving, European model lab |
| 80 | IBM | [developer.ibm.com/blogs](https://developer.ibm.com/blogs/) | watsonx, Granite, enterprise RAG, mainframe+AI, governance |
| 81 | Grafana Labs | [grafana.com/blog](https://grafana.com/blog/) | LGTM stack, LLM observability, Mimir/Loki/Tempo at planet scale |
| 82 | Confluent | [confluent.io/blog](https://www.confluent.io/blog/) | Kafka, Flink, streaming for agents, data contracts, exactly-once |
| 83 | HashiCorp | [hashicorp.com/blog](https://www.hashicorp.com/blog) | Terraform, Vault for agents, Consul, identity/secrets for machine users |
| 84 | Twilio | [twilio.com/blog](https://www.twilio.com/blog/) | Voice/SMS agents, Segment CDP, ConversationRelay, real-time comms |
| 85 | Adobe | [blog.developer.adobe.com](https://blog.developer.adobe.com/) | Firefly, Sensei, Creative Cloud AI, document intelligence, PDF at scale |
| 86 | Bloomberg | [bloomberg.com/company/stories](https://www.bloomberg.com/company/stories/) | BloombergGPT, financial NLP, market-data systems, terminal-scale infra |
| 87 | Riot Games | [technology.riotgames.com](https://technology.riotgames.com/) | Matchmaking, anti-cheat, live-ops, 150M+ player game services |
| 88 | Mozilla | [hacks.mozilla.org](https://hacks.mozilla.org/) | Firefox, Gecko, WASM, on-device AI, privacy-preserving ML |
| 89 | The New York Times | [open.nytimes.com](https://open.nytimes.com/) | Recsys, newsroom tools, content understanding, ads, privacy |
| 90 | Palantir | [blog.palantir.com](https://blog.palantir.com/) | AIP agents, ontology, Foundry, production decision systems |
| 91 | Capital One | [capitalone.com/tech](https://www.capitalone.com/tech/) | Bank-scale ML platforms, fraud, explainability, synthetic data |
| 92 | Zillow | [zillow.com/tech](https://www.zillow.com/tech/) | Home-value models, personalized search, CV for listings, model serving |
| 93 | Stitch Fix | [multithreaded.stitchfix.com](https://multithreaded.stitchfix.com/) | Algorithms org: recsys, inventory, demand, model monitoring |
| 94 | Expedia Group | [medium.com/expedia-group-tech](https://medium.com/expedia-group-tech) | Travel ranking, NLP inference, real-time user analytics |
| 95 | Bumble | [medium.com/bumble-tech](https://medium.com/bumble-tech) | Content moderation at scale, matching ML, image captioning |
| 96 | Algolia | [algolia.com/blog](https://www.algolia.com/blog/) | Neural / hybrid search, vector + keyword, RAG retrieval, ranking |
| 97 | Honeycomb | [honeycomb.io/blog](https://www.honeycomb.io/blog/) | Observability for LLM products, tracing, the “hard stuff” of shipping LLMs |
| 98 | Redis | [redis.io/blog](https://redis.io/blog/) | Redis Vector, semantic cache, session/memory for agents, low-latency serving |
| 99 | Pinecone | [pinecone.io/blog](https://www.pinecone.io/blog/) | Production vector DB, hybrid search, rerank, RAG architecture patterns |
| 100 | Cohere | [cohere.com/blog](https://cohere.com/blog) | Command/Rerank/Embed, enterprise RAG, retrieval research, serving |

---

## Same 100, copy-paste compact form

```
| Meta                    | https://engineering.fb.com/                                      | AI infra, tribal-knowledge agents, capacity agents, FAISS/PyTorch |
| Meta AI                 | https://ai.meta.com/blog/                                        | Llama, recsys, multimodal research into products                  |
| Google Cloud            | https://cloud.google.com/blog                                    | Agent Executor, Agent Platform, attested computations, MCP        |
| Google Research         | https://research.google/blog/                                    | Gemini, Pathways, long-context, TPU serving                       |
| Google DeepMind         | https://deepmind.google/discover/blog/                           | Frontier models, agents, Alpha-series                             |
| AWS Architecture        | https://aws.amazon.com/blogs/architecture/                       | AgentCore, graduated autonomy, multi-agent clusters               |
| AWS Machine Learning    | https://aws.amazon.com/blogs/machine-learning/                   | Bedrock agents, RAG, SageMaker, AWS Context                       |
| Amazon Science          | https://www.amazon.science/blog                                  | Alexa, retrieval, GNN, AutoML                                     |
| Netflix                 | https://netflixtechblog.com/                                     | Agentic causal inference, data canaries, live ops, recsys         |
| Uber                    | https://eng.uber.com/                                            | Agentic SDLC, MCP gateway, AI Context Graph, identity for agents  |
| LinkedIn                | https://www.linkedin.com/blog/engineering                        | QA Agent, AI code review, search stack, GNNs                      |
| Stripe                  | https://stripe.dev/blog                                          | Minions coding agents, ledger, auto-remediation, knowledge AI     |
| X / Twitter             | https://blog.x.com/engineering/en_us                             | Stale — last real eng posts ~2023                                 |
| Discord                 | https://discord.com/blog                                         | Cost attribution, voice outage, tracing                           |
| Dropbox                 | https://dropbox.tech/                                            | Nova coding-agent platform, cookie auditor, MCP                   |
| Databricks              | https://www.databricks.com/blog                                  | AI incident investigation, MLflow agent traces, Mosaic            |
| GitHub                  | https://github.blog/engineering/                                 | Copilot agents, stacked PRs, review cost                          |
| Shopify                 | https://shopify.engineering/                                     | River agent, gisting, Flow-from-language                          |
| Airbnb                  | https://medium.com/airbnb-engineering                            | LLM eval speed, auth, forecast unlearning                         |
| Canva                   | https://www.canva.dev/blog/engineering/                          | Session revocation at scale, design-AI                            |
| Lyft                    | https://eng.lyft.com/                                            | Localization, feature store, metrics layer                        |
| Pinterest               | https://medium.com/pinterest-engineering                         | Retrieval, resource provisioner, foundation models                |
| Zomato                  | https://blog.zomato.com/                                         | Gavel eval infra, real-time ads Flink                             |
| Spotify                 | https://engineering.atspotify.com/                               | Background coding agents, context layer, agentic DX               |
| Hotstar                 | https://blog.hotstar.com/                                        | 24/7 live-sports watchdog, CDN traffic shaping                    |
| Apple ML                | https://machinelearning.apple.com/                               | On-device LLMs, Apple Intelligence, private compute               |
| Microsoft Engineering   | https://devblogs.microsoft.com/engineering-at-microsoft/         | Copilot, identity, Windows/Azure AI                               |
| Microsoft Azure         | https://azure.microsoft.com/blog/                                | AI Foundry, Agent Service, confidential inference                 |
| NVIDIA                  | https://developer.nvidia.com/blog/                               | TensorRT-LLM, NIM, CUDA, NeMo, RAG blueprints                     |
| OpenAI                  | https://openai.com/news                                          | Agents SDK, Structured Outputs, Codex, evals                      |
| Anthropic               | https://www.anthropic.com/engineering                            | Context engineering, tools, Agent Skills, containment             |
| Hugging Face            | https://huggingface.co/blog                                      | PEFT/TRL, TGI, smolagents, serving                                |
| Slack                   | https://slack.engineering/                                       | Search, recsys, real-time messaging                               |
| Cloudflare              | https://blog.cloudflare.com/                                     | Workers AI, edge inference, AI Gateway                            |
| Figma                   | https://www.figma.com/blog/engineering/                          | Agentic SecOps, CRDTs, AI design features                         |
| PayPal                  | https://medium.com/paypal-tech                                   | Graph ML, fraud, XAI, payments scale                              |
| DoorDash                | https://careersatdoordash.com/engineering-blog/                  | Agent platform, forecasting, logistics ML                         |
| Snowflake               | https://www.snowflake.com/en/blog/                               | Cortex agents, governed LLM access                                |
| Salesforce              | https://engineering.salesforce.com/                              | Agentforce, Einstein, Hyperforce                                  |
| eBay                    | https://innovation.ebayinc.com/                                  | Search, ranking, catalog understanding                            |
| MongoDB                 | https://www.mongodb.com/blog/channel/engineering-blog            | Atlas Vector Search, distributed storage                          |
| Instacart               | https://tech.instacart.com/                                      | Real-time ML, marketplace balancer                                |
| Notion                  | https://www.notion.so/blog/topic/engineering                     | Notion AI, CRDT, workspace search                                 |
| Etsy                    | https://www.etsy.com/codeascraft                                 | Search ranking, ML platform                                       |
| Grab                    | https://engineering.grab.com/                                    | Data mesh, SuperApp, routing, ML platform                         |
| Atlassian               | https://www.atlassian.com/blog/atlassian-engineering             | Rovo agents, multi-product identity                               |
| Coinbase                | https://www.coinbase.com/blog/landing/engineering                | CEEcil memory agent, crypto infra                                 |
| Flipkart                | https://blog.flipkart.tech/                                      | HA MySQL, ZooKeeper, India-scale marketplace                      |
| Razorpay                | https://engineering.razorpay.com/                                | Payments auth, idempotency, risk ML                               |
| Reddit                  | https://www.reddit.com/r/RedditEng                               | Feed ranking, safety models, traffic spikes                       |
| Snap                    | https://eng.snap.com/                                            | On-device ML, AR, ads ranking                                     |
| Swiggy                  | https://bytes.swiggy.com/                                        | Hyperlocal search, routing, bandits                               |
| Walmart Global Tech     | https://medium.com/walmartglobaltech                             | ML platform, routing, omnichannel recsys                          |
| Booking.com             | https://booking.ai/                                              | Ranking, causal inference, experimentation                        |
| Grammarly               | https://www.grammarly.com/blog/engineering/                      | GEC, LLM rewriting, latency NLP                                   |
| Datadog                 | https://www.datadoghq.com/blog/engineering/                      | Bits AI, APM, observability agents                                |
| Vercel                  | https://vercel.com/blog/category/engineering                     | AI SDK, Fluid compute, agent credentials                          |
| Nubank                  | https://building.nubank.com/engineering/                         | Agents for 130M customers, foundation models                      |
| Klarna                  | https://engineering.klarna.com/                                  | Support agents, payments LLM ops                                  |
| Mercado Libre           | https://medium.com/mercadolibre-tech                             | ML platform, LatAm marketplace + fintech                          |
| Intuit                  | https://medium.com/intuit-engineering                            | GenOS, tax/finance agents                                         |
| Yelp                    | https://engineeringblog.yelp.com/                                | Local search, reviews NLP                                         |
| Twitch                  | https://blog.twitch.tv/                                          | Live video, chat fan-out, safety ML                               |
| Docker                  | https://www.docker.com/blog/                                     | Container isolation for agents, BuildKit                          |
| Postman                 | https://blog.postman.com/                                        | API agents, MCP, API governance                                   |
| GitLab                  | https://about.gitlab.com/blog/                                   | Duo agents, CI, code search                                       |
| Elastic                 | https://www.elastic.co/blog/                                     | Hybrid search, ELSER, vector + BM25                               |
| Cockroach Labs          | https://www.cockroachlabs.com/blog/                              | Distributed SQL, multi-region consensus                           |
| Sentry                  | https://blog.sentry.io/                                          | Seer AI debugging, tracing                                        |
| Block (Square)          | https://engineering.block.xyz/blog                               | Autohealing Moneybot, payments                                    |
| Duolingo                | https://blog.duolingo.com/hub/engineering/                       | LLM tutors, Birdbrain, production agents                          |
| NAVER                   | https://d2.naver.com/                                            | HyperCLOVA, search, DEVIEW                                        |
| Alibaba Cloud           | https://www.alibabacloud.com/blog                                | Qwen, PAI, ecommerce ranking                                      |
| TikTok / ByteDance      | https://developers.tiktok.com/blogs                              | Recsys, live, creator infra                                       |
| Mercari                 | https://engineering.mercari.com/en/blog/                         | C2C marketplace, Merpay                                           |
| Gojek                   | https://www.gojek.io/blog                                        | SuperApp, GoPay, ML platform                                      |
| Freshworks              | https://medium.com/freshworks-engineering-blog                   | Freddy AI, multi-tenant SaaS                                      |
| Zerodha                 | https://zerodha.tech/                                            | Tiny-team brokerage at India scale                                |
| Mistral AI              | https://mistral.ai/news                                          | MoE LLMs, open-weight serving                                     |
| IBM                     | https://developer.ibm.com/blogs/                                 | watsonx, Granite, enterprise RAG                                  |
| Grafana Labs            | https://grafana.com/blog/                                        | LGTM, LLM observability                                           |
| Confluent               | https://www.confluent.io/blog/                                   | Kafka/Flink for agent data planes                                 |
| HashiCorp               | https://www.hashicorp.com/blog                                   | Vault/Terraform for machine identity                              |
| Twilio                  | https://www.twilio.com/blog/                                     | Voice/SMS agents, Segment CDP                                     |
| Adobe                   | https://blog.developer.adobe.com/                                | Firefly, Sensei, document intelligence                            |
| Bloomberg               | https://www.bloomberg.com/company/stories/                       | BloombergGPT, market-data NLP                                     |
| Riot Games              | https://technology.riotgames.com/                                | Matchmaking, anti-cheat, live-ops                                 |
| Mozilla                 | https://hacks.mozilla.org/                                       | On-device AI, Gecko, WASM                                         |
| The New York Times      | https://open.nytimes.com/                                        | Recsys, newsroom tools                                            |
| Palantir                | https://blog.palantir.com/                                       | AIP agents, ontology, Foundry                                     |
| Capital One             | https://www.capitalone.com/tech/                                 | Bank ML platforms, fraud, XAI                                     |
| Zillow                  | https://www.zillow.com/tech/                                     | Home-value models, listing CV                                     |
| Stitch Fix              | https://multithreaded.stitchfix.com/                             | Recsys, inventory, model monitoring                               |
| Expedia Group           | https://medium.com/expedia-group-tech                            | Travel ranking, NLP inference                                     |
| Bumble                  | https://medium.com/bumble-tech                                   | Moderation, matching ML                                           |
| Algolia                 | https://www.algolia.com/blog/                                    | Neural/hybrid search, RAG retrieval                               |
| Honeycomb               | https://www.honeycomb.io/blog/                                   | Observability of LLM products                                     |
| Redis                   | https://redis.io/blog/                                           | Vector, semantic cache, agent memory                              |
| Pinecone                | https://www.pinecone.io/blog/                                    | Production vector DB, hybrid RAG                                  |
| Cohere                  | https://cohere.com/blog                                          | Command/Rerank/Embed, enterprise RAG                              |
```

---

## Read order if you care about LLM application logic

1. **Agent runtime / tools / context** — Anthropic Engineering, Uber, Shopify, Stripe, OpenAI News, Google Cloud, AWS ML, Figma, Coinbase, Nubank, DoorDash, Vercel
2. **Retrieval / ranking / RAG at product scale** — Pinterest, LinkedIn, Spotify, Airbnb, Elastic, Algolia, Pinecone, Etsy, Instacart, Booking.ai, NAVER D2
3. **Serving / inference / cost** — NVIDIA, Hugging Face, Cloudflare, Meta, AWS, Databricks, Apple ML
4. **Evals / observability / production failure** — Honeycomb, Datadog, Sentry, Grafana, Airbnb, Zomato (Gavel), Databricks (MLflow traces)
5. **Payments / ledgers / correctness** — Stripe, Block, Razorpay, Adyen-adjacent Adyen knowledge hub is extra: [adyen.com/knowledge-hub](https://www.adyen.com/knowledge-hub), PayPal, Coinbase
6. **Marketplace / logistics graphs** — Uber, Lyft, DoorDash, Grab, Gojek, Swiggy, Hotstar, Netflix

---

## Extra high-signal product blogs (overflow, not counted in the 100)

| Company | URL | Note |
| ------- | --- | ---- |
| LangChain | [blog.langchain.com](https://blog.langchain.com/) | LangGraph, context engineering, agent production |
| Adyen | [adyen.com/knowledge-hub](https://www.adyen.com/knowledge-hub) | Payments + applied AI research engineering |
| Robinhood | [robinhood.engineering](https://robinhood.engineering/) | Brokerage, market-data, reliability |
| Fastly | [fastly.com/blog](https://www.fastly.com/blog/) | Edge, WASM, traffic, AI at the edge |
| DigitalOcean | [digitalocean.com/blog](https://www.digitalocean.com/blog) | Gradient GPU cloud, inference products |
| Supabase | [supabase.com/blog](https://supabase.com/blog) | pgvector, auth, realtime, AI tooling |
| PlanetScale | [planetscale.com/blog](https://planetscale.com/blog) | Vitess, serverless MySQL at scale |
| Scale AI | [scale.com/blog](https://scale.com/blog) | Data engines, evals, enterprise agents |
| xAI | [x.ai](https://x.ai/) | Grok model drops (thin eng blog) |
| LMSYS | [lmsys.org/blog](https://lmsys.org/blog/) | Arena, serving research (Vicuna/SGLang lineage) |

---

## Communities that surface *new* posts from these blogs

| Place | URL |
| ----- | --- |
| r/LLMDevs | [reddit.com/r/LLMDevs](https://www.reddit.com/r/LLMDevs/) |
| r/Rag | [reddit.com/r/Rag](https://www.reddit.com/r/Rag/) |
| r/LocalLLaMA | [reddit.com/r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) |
| r/MachineLearning | [reddit.com/r/MachineLearning](https://www.reddit.com/r/MachineLearning/) |
| HN (eng-blog threads) | [news.ycombinator.com](https://news.ycombinator.com/) |
| Uber Engineering on X | [x.com/UberEng](https://x.com/UberEng) |
| Shopify Engineering on X | [x.com/ShopifyEng](https://x.com/ShopifyEng) |
| Netflix Tech on X | [x.com/NetflixEng](https://x.com/NetflixEng) |

Recent public signal used while compiling (Aug–Sep 2026): Uber’s agent-cost post (7× weekly agent users, MCP gateway, Context Graph), Shopify gisting, Coinbase CEEcil memory agent, Google Cloud Next 2026 Agent Platform, AWS AgentCore / AWS Context, Figma agentic SecOps, Nubank agents for 130M customers.

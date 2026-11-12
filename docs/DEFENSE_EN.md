# Defence pack for the admissions panel (PolyU · CityU · HKU · HKUST)

Written to be **learned and delivered by you**, in English. It is deliberately more cautious than the first draft you were given:
every claim below is something the repository can show on screen, and the weak points are named so that you name them first.

---

## 0. The one-sentence version
> "I built a tool that lets a city test how safe its streets are *before* anyone gets hurt — a rule-abiding driving agent drives thousands of trips through a real district, finds where it is repeatedly forced into near-misses, and ranks those places on a map."

## 1. Why it matters (the problem — with sources)
* **Hong Kong:** in 2024 police recorded 2,665 pedestrian casualties and 42 pedestrian deaths out of 88 traffic fatalities — almost half of all deaths were people on foot ([Hong Kong Police, *Traffic Report 2024*](https://www.police.gov.hk/info/doc/statistics/traffic_report_2024_en.pdf)). Elderly pedestrians are over-represented in fatal and serious accidents ([The Standard](https://www.thestandard.com.hk/news/article/329630/Over-half-of-traffic-fatalities-pedestrians-most-elderly-authorities-say)). Hong Kong also publishes accident statistics as open data ([data.gov.hk](https://data.gov.hk/en-data/dataset/hk-td-wcms_18-road-traffic-accident-stat-2024/resource/935b1317-463f-42a3-98ea-4b4a044d6805)), which is exactly what a validation study needs.
* **Autonomous vehicles are arriving in Hong Kong's streets now:** the Transport Department has issued pilot licences for autonomous vehicles under Cap. 374AA ([TD press release, Aug 2025](https://www.info.gov.hk/gia/general/202508/14/P2025081400789p.htm); [Digital Policy Office](https://digitalpolicy.gov.hk/en/our_work/success_stories/autonomous_vehicle_pilot_licence)). Cities will need cheap ways to ask *"is this junction ready for them — and for pedestrians?"*
* **The gap:** crashes are rare and tragic, so planners learn from them too late. Traffic engineers therefore use *surrogate safety measures* — near-misses such as low time-to-collision or hard braking — that can be measured long before a crash. The method is established ([FHWA surrogate-safety research](https://www.fhwa.dot.gov/publications/research/safety/08049/08049.pdf)) but its predictive validity is still debated ([systematic review of 549 studies, QUT](https://eprints.qut.edu.au/209080)). So I treat my output as a **screening tool that generates hypotheses**, not as a crash predictor.

## 2. What exactly I built
1. A pipeline that turns any OpenStreetMap district into a multi-level simulated city (tunnels, flyovers, signals, zebras, buses, pedestrians). Two real districts of Astana are included.
2. A learned driving agent trained with behaviour cloning → DAgger → PPO, whose observation contains **no map coordinates**, so it cannot memorise a district; it is evaluated on districts it has never seen.
3. **Rule-first metrics:** violations counted per rule, "justified vs. needless" braking, how often the emergency brake had to rescue the policy.
4. **The audit:** the agent as a probe → near-miss events with GPS positions → heat-map and ranked hot-spots → *what-if* comparison (speed limit ×0.8, longer green, aggressive or calm other road users) on identical trips with bootstrap confidence intervals.
5. One product: a Studio window and a 3-D digital twin that explains the car's decisions on screen.

## 3. "Isn't this just Tesla?" — the honest answer
No, and I do not compete with Tesla. A car company asks *"how does my vehicle see and drive?"* I ask *"is this place safe for the people who use it?"*. The driver is the **measuring instrument**; the product is the **audit**. What is genuinely new here is the *combination* and the accessibility: any OSM district in about a minute, multi-level roads, rule-based metrics, an explanation overlay, running on one consumer GPU (an RTX 4060 laptop). PPO, DAgger, IDM and surrogate measures are existing techniques — I say that myself.

## 4. What I can show — and what I will not claim
| I can show (live) | I will not claim |
|---|---|
| Real EXPO / Nurzhol districts in 3-D with flyovers, signals, pedestrians | That it predicts crashes or replaces engineers |
| Rule counters, decision text, emergency-brake reliance in the HUD | That the network is "safe" before I show evaluation numbers |
| Audit: heat-map, ranked junctions with GPS, what-if table with intervals | That hot-spots match real crash sites (not yet validated) |
| Expert baseline ≈ 68–75 % success on real Astana, 0 rule violations | A trained-network result I have not measured |
| 88 automated tests; simulation on its own thread, ~137 fps measured on the laptop | That it works on real vehicles |

**Corrections to common overstatements** (do not say these): "10,000 agents with many driving styles" — the audit uses hundreds to thousands of trips of one probe policy; *other road users* can be switched between calm and aggressive. "I implemented F = ma physics" — the vehicle is a kinematic bicycle model with actuator lag; say exactly that. "Ultimate proof of engineering maturity" — let the repository speak instead.

## 5. The 90-second pitch (learn this)
> "Hello, I'm ___ from Kazakhstan. Road injuries are a leading cause of death worldwide, and in Hong Kong almost half of last year's traffic deaths were pedestrians. Planners usually learn that a junction is dangerous after someone is hurt.
>
> I built **Astana Pilot**. It takes a real district from OpenStreetMap and creates a simulated city with the real layout — flyovers, traffic lights, zebra crossings, buses and pedestrians. Then a self-driving agent drives thousands of trips. It never sees map coordinates, so it has to generalise to streets it has not memorised.
>
> What makes it useful is what I measure. I count every traffic-rule violation separately. I distinguish braking for a reason from needless braking. And I log near-misses with GPS positions, so the system produces a risk heat-map and a ranked list of junctions. I can also ask what-if questions — for example, what a 20 percent lower speed limit changes — on identical trips, with confidence intervals.
>
> It runs on my laptop GPU, and the same tool works for any city, including Hong Kong. The honest limits: it is a simulation, the risk ranking is a screening tool that still needs validation against real accident data — Hong Kong publishes that data, which is exactly my next step — and I currently support right-hand traffic only. I used an AI coding assistant as a tool; the problem, the evaluation design and the conclusions are mine, and I can walk you through any part of the code. Thank you."

## 6. Five-minute live demo (Studio → everything on one screen)
1. **Studio:** "One window: choose a district, choose a driver — expert or my trained network." (20 s)
2. **Start 3-D simulation** on EXPO: point at the speed cluster, the rule chip ("all clear"), the radar and the decision line. Press `C` (cockpit) and `T` (dusk). (60 s)
3. **Switch to Nurzhol:** flyover and ramps — "each road level has its own collision field". (30 s)
4. **Run audit** (pre-computed results are shown in the Studio): heat-map, top-5 junctions, what-if table. "This is the product." (90 s)
5. **Evaluation slide:** seen vs. unseen district, violations per rule, with the honest weak spots. (60 s)
6. **Next steps:** left-hand traffic, validation against Hong Kong accident data, camera perception. (30 s)
Have the 60-second video as a fallback.

## 7. Questions you will get (and short, honest answers)
1. **Why this project?** Safety audits need data that arrives only after harm; simulation can supply an early signal. It also lets me study reinforcement learning on a real, meaningful task.
2. **What is the contribution beyond existing tools?** The combination — OSM-to-multilevel-twin in a minute, rule-first metrics, zero-shot district evaluation, near-miss audit with what-if and intervals — on consumer hardware. I do not claim new algorithms.
3. **How do you know the audit is right?** I don't yet. Surrogate measures correlate with crashes in some studies but their validity is debated. Plan: compare my ranking with Hong Kong's open accident records and report rank correlation, including negative results.
4. **Why would a *learned* agent be better than a rule-based one for auditing?** It may expose failure modes a hand-written driver never meets, and it shows how an AV would experience a junction. The rule-based expert is already usable as a probe; comparing both is an experiment.
5. **Sim-to-real?** Not claimed. Missing: cyclists, sensor noise, camera perception. Domain randomisation (friction, signal phase, traffic) is a first step.
6. **How is generalisation tested?** Observations contain no absolute positions; I evaluate on districts and synthetic cities never used in training, with Wilson intervals.
7. **What if the agent just relies on the emergency brake?** The AEB share is reported separately, so that would be visible.
8. **Hardest bug?** The expert drove into kerbs on narrow one-way streets (46 % → 68 % success after limiting swerves by road width). Another: all cars drove sideways because Panda3D's heading is mirrored in a left-handed frame — I wrote a regression test that checks the orientation numerically.
9. **Did you write this alone?** I specified the problem, the metrics and the experiments, reviewed and tested the work, and used an AI assistant for most of the implementation. I can explain each module — pick one.
10. **Your grades in physics/maths?** *(Answer only if asked; do not volunteer.)* "They are not where I want them, and this project is part of how I am fixing that: I had to learn the maths behind it — kinematics, car-following models, distance transforms, policy gradients — and my SAT Math score is 770." Keep it short, then return to the work.
11. **Why PolyU / CS?** Name one lab or course you actually checked (e.g. smart mobility, geomatics, AI), and say what you would do there: *validate the audit with Hong Kong data and add left-hand traffic.*
12. **Ethics?** Simulation only; no personal data (OSM and public statistics); risk of over-trusting a model — hence the explicit limitations and the screening-not-prediction framing.
13. **What would you do with one more month?** Left-hand traffic and a Hong Kong district; validation study; cyclists.
14. **Explain PPO in one minute.** The policy is updated in small, clipped steps so that a new policy never strays too far from the old one; the clip keeps training stable.
15. **Explain DAgger.** The student drives, the expert labels what it would have done in the states the student actually visits, which fixes the distribution shift of plain imitation.

## 8. Make it truly yours — preparation plan (3 weeks)
* **Week 1:** run the 8-hour training; record numbers in `docs/RESULTS.md`; read `docs/ARCHITECTURE.md`; explain each module aloud in one sentence.
* **Week 2:** three personal experiments (no-BC ablation; train on EXPO → test on Nurzhol; audit variants `aggr` vs `calm`); write the findings in `docs/DEV_LOG.md` in your own words.
* **Week 3:** rehearse the 90-second pitch and the demo ten times; practise Q&A with a friend; prepare a one-page poster; check two Hong Kong faculty pages and cite one paper accurately.
* Stretch (high value): download Hong Kong's open accident data and run the ranking validation on a Hong Kong district — it turns a demo into research.

## 9. Disclosure line for your application
"Developed with an AI coding assistant under my direction; I designed the problem, the evaluation protocol and the experiments, and I take responsibility for all claims. Every module has automated tests."

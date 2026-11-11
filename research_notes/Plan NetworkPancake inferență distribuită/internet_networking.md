# Internet networking for distributed LLM inference mesh (phase 2) after LAN phase (phase 1)

Note: research done 2026-11-10 with limited tool budget. Several items (iOS/Android client details of Tailscale/ZeroTier, libp2p mobile, per-token RTT math) are partly inference; gaps are listed per section.

## 1. Transport / NAT traversal options (Tailscale, Headscale, ZeroTier, WireGuard, WebRTC, libp2p, QUIC/iroh, port forwarding, relay)

### Takeaway
Direct UDP hole punching works for roughly 70-90% of peer pairs depending on measurement (Tailscale author estimates >90% direct; libp2p DCUtR measured ~70% at the hole-punch stage in the wild); the remainder (symmetric/CGNAT on both sides, UDP-blocking networks) needs a relay, so any design must include a relay fallback. For non-technical users, an overlay (Tailscale, or Headscale for self-hosting) is the least effort; embedding iroh or libp2p in the app gives the best integration but costs engineering.

### Cited Findings
- Tailscale author estimates implementing the core NAT-traversal techniques yields "a direct connection over 90% of the time"; DERP relays guarantee connectivity otherwise at extra latency and possibly lower bandwidth; DERP runs over HTTP so it works on strict networks. — [Tailscale: how NAT traversal works](https://tailscale.com/blog/how-nat-traversal-works)
- Hard (endpoint-dependent mapping / symmetric) NAT: one hard NAT on one side can be beaten with the birthday-paradox trick (256 open ports on the easy side: 174 probes ~50%, 1,024 probes 98%, 2,048 probes 99.9%); two hard NATs is essentially hopeless (~0.01% after 20 s; 99.9% would need ~170,000 probes per side, ~28 min). — [Tailscale blog](https://tailscale.com/blog/how-nat-traversal-works)
- CGNAT = double NAT; port-mapping protocols (UPnP/NAT-PMP/PCP) often disabled or broken under it; hairpinning often unsupported, so "you're stuck with relaying". Some networks (e.g. guest Wi-Fi allowing only DNS) block all outbound UDP. UDP firewall idle timeouts ~30 s (keepalives needed). IPv6 adoption cited as ~33%. — [Tailscale blog](https://tailscale.com/blog/how-nat-traversal-works)
- libp2p DCUtR large-scale measurement (IMC '26): >4.4 M traversal attempts, >85,000 networks, 167 countries; conditional success of hole-punching stage 70% +- 7.1%; TCP and QUIC statistically indistinguishable; independent ProbeLab honeypot: 82.9% for Port-Restricted Cone class, aggregate consistent with ~70%. Earlier talks: ~72% (2022), ~80% for Go clients later, lower for VPN clients. — [ProbeLab summary of the paper](https://probelab.io/publications/large-scale-measurement-of-nat-traversal-for-the-decentralized-web-a-case-study-of-dcutr-in-ipfs/); [arXiv 2510.27500](https://arxiv.org/html/2510.27500v1); [ProbeLab blog](https://probelab.io/blog/can-libp2p-punch-through-nats/)
- libp2p circuit relay v2 relays are deliberately limited (per-connection duration and bytes-per-direction caps set by relay operator; 0 = unlimited; relay resets streams past the cap), designed to only bootstrap hole punching, not carry bulk traffic. Implication: public libp2p relays are unsuitable as a data path for inference unless you run your own with no limits. — [libp2p circuit-v2 spec](https://github.com/libp2p/specs/blob/master/relay/circuit-v2.md)
- iroh: QUIC (TLS 1.3) connections with hole punching done inside QUIC (`n0_nat_traversal` extension), relay fallback, seamless upgrade from relay to direct path without dropping the connection, Swift and Kotlin bindings; no success-rate numbers published on that page. — [iroh docs](https://docs.iroh.computer/what-is-iroh)
- Tailscale free "Personal" plan: up to 6 users, unlimited user devices, 50 tagged resources; paid Standard $8/user/mo, Premium $18/user/mo. — [Tailscale pricing](https://tailscale.com/pricing)
- ZeroTier free Personal plan: 10 devices on one network, one admin, personal non-commercial use; Essential $18/mo (10 devices, +$2/device), Scale $179/mo (100 devices). — [ZeroTier pricing](https://www.zerotier.com/pricing/)
- Headscale: open-source (BSD-3-Clause) self-hosted Tailscale control server; supports a single tailnet only; scoped to personal/small orgs; ~45.5k stars, actively developed, presented at FOSDEM 2026. Official Tailscale clients can point at it. — [Headscale GitHub](https://github.com/juanfont/headscale)
- WebRTC TURN: vendor blogs estimate 10-25% of sessions need TURN (no methodology; treat as rough); Cloudflare Realtime TURN $0.05/GB with 1,000 GB free; Twilio ~ $0.40/GB and self-hosted coturn egress $0.02-0.09/GB per secondary sources (unverified). Symmetric/CGNAT on mobile carriers frequently forces TURN. — [Cloudflare TURN docs](https://developers.cloudflare.com/realtime/turn/faq/); [bloggeek TURN glossary](https://bloggeek.me/webrtcglossary/turn/); [ExpressTURN on mobile NAT](https://blog.expressturn.com/webrtc-calls-fail-on-mobile-networks)

### Inferences
- Comparison summary (qualitative, from above plus general knowledge; not all verified):
  - Tailscale: lowest user effort (install app, log in; iOS/Android apps exist), >90% direct, DERP fallback; free tier fits "own/family devices" (6 users) but not an open volunteer mesh (each volunteer would need to be in your tailnet or share nodes; ToS/limits). Good for phase 2 trusted-peers mode.
  - Headscale: same clients, removes vendor dependence and user limits, but you operate the control plane + DERP; single tailnet; operator burden.
  - ZeroTier: similar, 10-device free cap, non-commercial restriction; weaker mobile-carrier traversal reputation (not verified here).
  - Raw WireGuard: efficient but manual key/endpoint management; no built-in traversal -> only with a coordinator; not for non-technical users.
  - WebRTC data channels: only option usable from a browser/PWA; ICE/STUN/TURN standard; SCTP-over-DTLS throughput and head-of-line behavior less ideal for bulk tensors; TURN costs apply for relayed share.
  - libp2p: full P2P stack (Noise/TLS, DCUtR, relays, AutoNAT); ~70% hole punching; engineering heavy, mobile support uneven (not verified).
  - iroh/QUIC: modern, single-library, QUIC multiplexing and connection migration suit mobile; run your own relay (open source) for the fallback.
  - Plain TCP + port forwarding: unacceptable for non-technical users, fails under CGNAT; UPnP unreliable.
  - Relay server: always works, but adds a hop of latency and your bandwidth bill; should be fallback not primary.
- Because a ~10-30% relayed share is realistic, budget for relay bandwidth: activations per token are small (see section 2), so relay cost is modest (hidden-state vectors of a few KB per token per hop).

### Gaps
- No verified numbers for Tailscale/ZeroTier throughput limits, WireGuard-over-DERP throughput, or iOS/Android battery/background constraints (VPN-extension memory limit on iOS, etc.). Not found in this pass.
- No independent measure of carrier-CGNAT symmetric NAT share; vendor numbers only.
- libp2p/iroh on iOS/Android maturity not verified.
- Tailscale ToS on non-personal/volunteer-mesh use not checked.

## 2. Latency reality, per-token floor, hiding techniques, Petals numbers

### Takeaway
Pipeline decode needs one network hop per stage per token, so tokens/s is bounded by 1/(sum of hop RTTs + compute); with tens-of-ms home links this caps single-stream decode at roughly 1-6 tok/s over many stages, which matches Petals (~1 step/s for BLOOM-176B in real world; up to 4-6 tok/s claimed for large Llama/Falcon). Hiding requires speculative decoding, batching many concurrent sequences, and few pipeline stages (fewer, larger shards per device).

### Cited Findings
- Ookla Q4 2025 (US): mobile operators minimum latency ~27 ms, multi-server latency ~51 ms; fixed operators ~12 ms minimum and ~25 ms multi-server. — [Fierce Network citing Ookla](https://www.fierce-network.com/wireless/opinion-reality-check-ai-latency-30-ms-milestone)
- NetForecast (2023): cable median latency near 50 ms; 5G fixed wireless ~70-80 ms; LEO satellite ~90 ms (to test servers, loaded conditions). — [NetForecast report](https://www.netforecast.com/wp-content/uploads/FixedWireless_LEO_CableComparisonReport_NFR5148-1.pdf)
- Older Ofcom-based data: 4G RTT ~53 ms, 3G ~63.5 ms; fixed jitter depends on time of day/congestion. — [arXiv 1703.09434](https://arxiv.org/pdf/1703.09434)
- WonderNetwork single-pair snapshots (30-ping averages): Cincinnati-Jerusalem ~162-177 ms (2026); Copenhagen-Yangon ~216-230 ms; Yangon-Chicago ~263 ms. Intercontinental pairs are thus ~100-250+ ms. — [WonderNetwork pings](https://wondernetwork.com/pings)
- Petals paper (BLOOM-176B, batch 1, seq 128): 3 A100 servers 1 Gbit/<5 ms: 1.71 steps/s; 100 Mbit/100 ms: 1.23 steps/s; 12 virtual servers 100 Mbit/<5 ms: 1.24 steps/s, at 100 ms: 0.57 steps/s; 14 real servers in Europe/N. America: 0.83 steps/s (seq 2048: 0.79). Parallel forward (batch 64): 179 tokens/s real world vs 44 tokens/s in the 100 ms 12-server sim. Offloading 1xA100: 0.18 steps/s. — [Petals paper, arXiv 2209.01188](https://arxiv.org/html/2209.01188)
- Petals README: single-batch inference up to 6 tok/s Llama 2 70B and up to 4 tok/s Falcon 180B (project claim); earlier README cited 3-4 steps/s for LLaMA-65B and ~1 step/s BLOOM-176B ("up to 10x faster than offloading"), parallel inference hundreds of tokens/s. — [Petals GitHub](https://github.com/bigscience-workshop/petals); [PyPI README](https://pypi.org/project/petals/2.0.0)
- Petals compresses hidden states with dynamic blockwise 8-bit quantization before sending, which "halves the bandwidth requirements without any noticeable effect on generation quality"; clients need ~25 Mbit/s, servers >=100 Mbit/s. — [arXiv 2209.01188](https://arxiv.org/html/2209.01188)
- Decentralized Speculative Decoding (DSD): models cross-node cost ~ (N-1)*t1*(k-1)/k saved; reports up to 2.56x (HumanEval) and 2.59x (GSM8K) speedups, plus 15-20% from adaptive verification. — [arXiv 2511.11733](https://arxiv.org/abs/2511.11733)
- BloomBee (internet-scale): pipeline parallelism + lossless activation compression + speculative decoding; reports naive speculative decoding can hurt throughput on slow links because candidate tokens add traffic, mitigated by a learned classifier. — [arXiv 2604.21072](https://arxiv.org/pdf/2604.21072)

### Inferences
- Floor estimate: with S pipeline stages on separate homes, each token costs about S x (RTT/2 one-way ... ) in a ring; per token round trip through the chain ~ sum of one-way hop latencies plus return to client. E.g. 4 stages at 25 ms one-way homes ~ 100 ms -> <10 tok/s ceiling before compute; intercontinental hops (100+ ms one-way) -> 1-3 tok/s. Mobile jitter (RTT tens of ms variation) hurts tail latency more than mean.
- Consistent with Petals real-world 0.83 steps/s: Petals was RTT-dominated across 14 servers (many hops), and the sim shows 100 ms latency halves speed (1.24 -> 0.57).
- Hiding techniques ranked: (1) fewer stages / bigger shard per device and prefer geographically close peers (latency-aware routing, which Petals does); (2) speculative decoding (2-2.6x in DSD) since k tokens verified per round trip; (3) many concurrent sequences/micro-batches (Petals batch 64 gives ~5x tokens/s over batch 1 in real world: 179 vs 33 tokens/s) - helps throughput not single-user latency; (4) 8-bit/4-bit activation quantization; (5) keep all requests of a session pinned to same chain with KV cache on-node.
- Activation size per token is small (hidden dim x 2 bytes, e.g. ~8-16 KB for 4-8k hidden dims in fp16; half with int8), so bandwidth is not the bottleneck for decode, latency is; prefill (long prompts) is bandwidth-bound.

### Gaps
- No speedtest dataset with home-to-home same-country/continent RTT medians and mobile jitter percentiles found; only operator-level numbers (US-centric).
- Petals' tok/s figures for Llama 70B/Falcon 180B are README claims without a measurement table in what I retrieved; the project appears not to be heavily maintained (not verified; repo shows 94 open issues).
- DSD and BloomBee absolute latency settings not retrieved.

## 3. Security: peer authentication, encryption, privacy of activations, integrity, DoS, trusted-peers-first

### Takeaway
Activations are not private: first-layer peers can recover input tokens, and inversion attacks recover large fractions of tokens even from deeper splits. For a first release, only allow explicitly paired, authenticated peers over an encrypted overlay/transport (WireGuard/QUIC-TLS/Noise); treat open volunteer swarms as a later, separate mode with explicit privacy warnings and integrity checks.

### Cited Findings
- Petals paper: peers serving first layers "can use their inputs to recover input tokens"; recommends trusted servers or private swarm for sensitive data; servers may return incorrect outputs (malicious or lazy), proposed mitigation is staking/pledge points plus published hashes of inputs/outputs for verification. — [arXiv 2209.01188](https://arxiv.org/html/2209.01188)
- Petals README: public swarm data is processed by other people; suggests private swarm among trusted people. — [Petals GitHub](https://github.com/bigscience-workshop/petals)
- Activation inversion preprint (single author, not peer-reviewed): 3-layer MLP decoder recovers ~59% of tokens at a 2-layer split; ~35% with 8 local layers. — [arXiv 2602.16760](https://arxiv.org/html/2602.16760v1)
- iroh gives authenticated, encrypted connections via QUIC with TLS 1.3 (peer identity = public key). — [iroh docs](https://docs.iroh.computer/what-is-iroh)

### Inferences
- Layers: (a) transport encryption + peer auth: WireGuard (Tailscale/Headscale), QUIC+TLS1.3 with key-pinned peer IDs (iroh), or libp2p Noise; all give mutual auth if peer keys are exchanged out of band (QR pairing). (b) application-level allowlist of peer public keys. (c) Per-peer role limits (only compute shard, no filesystem access).
- Privacy cannot be fixed by encryption in transit; the compute peer sees plaintext activations. Mitigations: keep embedding and first/last N layers on the user's own device (reduces but does not remove leakage), trusted-peers only, or confidential-compute (not applicable to home PCs).
- Model integrity: verify weight shard hashes (SHA-256 manifest, signed) before loading; redundancy/spot-check challenges (duplicate a random request to a second peer and compare) for untrusted mode; reproducibility problems across different GPUs/quantization (floating-point nondeterminism) make exact equality checks unreliable - use tolerance.
- DoS: allowlist + per-peer rate limits and authenticated connections before allocating KV cache; relays need quotas.
- Recommendation: phase 2 v1 = trusted-peers-only (own devices + friends), keys exchanged via QR code; open mesh deferred.

### Gaps
- No found empirical study of verification overhead for volunteer inference; no formal evaluation of the Petals integrity proposal being implemented.
- Didn't evaluate side-channel/timing leaks or KV-cache leakage.

## 4. LAN discovery (mDNS/DNS-SD, UDP broadcast, QR pairing; Android/iOS constraints)

### Takeaway
mDNS/DNS-SD is the right primary mechanism on both platforms, but both OSes now gate local-network access behind permissions (iOS since 14; Android enforcement mandatory for apps targeting API 37). Always provide QR-code / manual IP fallback pairing.

### Cited Findings
- Android NsdManager implements DNS-SD: registerService (NsdServiceInfo with name, `_type._tcp`, port from ServerSocket(0)), discoverServices, resolveService; discovery is expensive and should be stopped on pause. — [Android NSD guide](https://developer.android.com/develop/connectivity/wifi/use-nsd)
- Android local network permission: opt-in in Android 16, mandatory for apps targeting Android 17 (API 37) or higher; `ACCESS_LOCAL_NETWORK` in NEARBY_DEVICES group; docs recommend system picker (`FLAG_SHOW_PICKER`) for device discovery without the broad permission. A secondary source claims mDNS/SSDP/NsdManager/.local resolution require it under Android 17 and gives Aug 2026 dates; not confirmed on official pages. — [Android local network permission](https://developer.android.com/privacy-and-security/local-network-permission); [xenospectrum (secondary)](https://xenospectrum.com/en/android-network-protection/)
- iOS local network privacy (iOS 14+) (from the model's general knowledge since the fetch returned only the page title; verify against the TN): prompt triggered by Bonjour browse/advertise or direct local unicast/multicast/broadcast; Info.plist needs `NSLocalNetworkUsageDescription` and `NSBonjourServices`; sending/receiving raw multicast requires `com.apple.developer.networking.multicast` entitlement granted by Apple on request; internet traffic exempt. — [Apple TN3179](https://developer.apple.com/documentation/technotes/tn3179-understanding-local-network-privacy) (page text not retrieved; claims unverified)

### Inferences
- Use Bonjour/NSD service `_pancake._tcp` with TXT record carrying peer-id fingerprint; avoids multicast entitlement on iOS because Bonjour via NWBrowser needs only the Info.plist keys (general knowledge, verify). Raw UDP broadcast on iOS requires the entitlement - avoid.
- Pairing: QR code carrying peer public key + one-time token + (optional) LAN IP/port and relay hint; works regardless of mDNS, client-isolated Wi-Fi/guest networks (AP isolation blocks mDNS), and aligns with the trusted-peers security model.
- Phones in background: OS restrictions on listening sockets mean phones are mostly clients/leaf compute nodes in phase 1.

### Gaps
- Apple TN3179 and multicast-entitlement text not actually retrieved; details above need verification.
- Android 17 NsdManager enforcement specifics unconfirmed.

## 5. Recommended minimal stacks

### Takeaway
Phase 1 (LAN): TCP or QUIC over LAN, mDNS discovery + QR pairing, pinned-key mutual TLS. Phase 2 (internet): trusted-peers overlay via Tailscale (or Headscale) first because it gives traversal, encryption and relays for free with minimal code; consider embedding iroh (QUIC + relay) as the long-term in-app option.

### Cited Findings
- (Supporting facts are in sections 1-4; no additional sources.)

### Inferences
- Phase 1: gRPC/raw TCP (or QUIC) between stages; mDNS/DNS-SD (`NsdManager`, Bonjour) + QR code pairing; each device has an Ed25519 identity key; connection authenticated with mutual TLS pinned to paired keys; 8-bit activation compression; no NAT work.
- Phase 2a (trusted peers, fastest): users install Tailscale (6-user free tier fits personal/family) or you host Headscale (BSD-3, single tailnet); app talks to 100.x addresses over the same transport as phase 1; discovery via tailnet peer list/MagicDNS; unchanged app security model; DERP relay fallback included. Caveats: requires users to install a second app (VPN slot conflict on phones: only one VPN active at a time on Android/iOS, general knowledge); free-tier limits.
- Phase 2b (in-app, no extra install): embed iroh (QUIC, hole punching, relay fallback, Swift/Kotlin bindings, public-key identities) with your own relay server(s); expect relayed share of maybe 10-30%. libp2p is an alternative if you need DHT/pubsub for an open swarm, with ~70% hole punching and limited relays (run your own).
- Scheduling for WAN: latency-aware stage assignment (cluster nearby peers, minimize hops, prefer 1-2 big stages), speculative decoding, batch concurrent requests, int8 activations, KV cache pinned to chain.
- Defer open volunteer mode until integrity checking and privacy policy designed.

### Gaps
- No benchmark of iroh vs Tailscale for sustained tensor throughput found; recommend measuring on target devices.
- Cost model for own relay bandwidth not computed (depends on user count).

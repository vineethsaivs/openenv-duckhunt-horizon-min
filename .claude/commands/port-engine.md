Port the Duck Hunt game engine from ~/horizon_min/duck_hunt_openenv/server/ 
into duckhunt_env/server/game_engine.py and duckhunt_env/server/renderer.py.

Key requirements:
1. DuckHuntGame class with: reset(), advance_frame(), shoot(x, y, advance_frames), 
   is_over(), score, ducks_remaining, bullets_remaining, misses
2. shoot() returns (result, base_reward, distance) where result is one of: 
   "hit", "double_kill", "miss", "no_target"
3. render_frames(game) returns a PIL Image (headless, no display)
4. Deterministic replay via snapshot + RNG seeds
5. Circle-based collision detection
6. Duck velocity/wall bouncing/random direction changes
Do NOT change game logic — just adapt the interface.

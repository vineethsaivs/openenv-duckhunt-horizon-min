Build the complete OpenEnv 0.2.1 server for Duck Hunt.

Create these files exactly:
1. duckhunt_env/client/models.py — DuckHuntAction and DuckHuntObservation Pydantic models
   (see CLAUDE.md for field specs)
2. duckhunt_env/server/duckhunt_environment.py — DuckHuntEnvironment class inheriting 
   from openenv.core.env_server.interfaces.Environment
3. duckhunt_env/server/app.py — using create_app(DuckHuntEnvironment)
4. duckhunt_env/client/duckhunt_client.py — HTTPEnvClient subclass

Reference the exact code from the battle plan PDF in the repo root.

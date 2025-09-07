# Repository Guidelines

## Project Structure & Modules
- Root Maven multi-module project: `cache/`, `runelite-api/`, `runelite-client/`, `runelite-jshell/`, `runelite-maven-plugin/`.
- RLBot plugin: `rlbot-plugin/` (standalone Maven module, sideloaded into RuneLite).
- Scripts: `run_runelite.sh`, `hot-reload-plugin.sh`, `rlbot-plugin/build-plugin.sh`.
- Upstream client (separate checkout) artifacts are expected under `runelite/runelite-client/target/`.

## Build, Test, Run
- Build all modules: `mvn -DskipTests clean install`
- Run tests: `mvn test`
- Build RLBot plugin: `./rlbot-plugin/build-plugin.sh`
- Hot-reload plugin (build + copy to sideload dirs): `./hot-reload-plugin.sh`
- Launch RuneLite (requires shaded JAR at `runelite/runelite-client/target/client-<ver>-shaded.jar`): `./run_runelite.sh`
- Java: use JDK 17 for running and building locally.

## Coding Style & Naming
- Language: Java (modules use Lombok).
- Indentation: 4 spaces; max line length guided by `checkstyle.xml`.
- Packages: `net.runelite.*`; plugin code under `net.runelite.client.plugins.rlbot`.
- Class naming: `PascalCase`; methods/fields `camelCase`; constants `UPPER_SNAKE_CASE`.
- Static analysis: Checkstyle and PMD (`runelite-client/pmd-ruleset.xml`). Fix warnings or justify in PR.

## Testing Guidelines
- Framework: JUnit 4 via Maven Surefire.
- Location: `src/test/java/...`; name tests `*Test.java`.
- Run module tests: `mvn -pl <module> test` (e.g., `-pl runelite-client`).
- Add targeted unit tests for new logic; no strict coverage gate, but prefer meaningful assertions over broad mocks.

## Commit & Pull Requests
- Commits: imperative mood, concise subject, optional scope (e.g., `rlbot-plugin:`). Example: `rlbot-plugin: fix task selection timing`.
- PRs must include:
  - Summary of change and rationale.
  - Linked issue (if applicable).
  - Testing notes and local run steps; screenshots/log snippets for UI/overlay changes.
  - Scope of impact (modules/files touched) and any config updates.

## Security & Configuration
- Sideloaded plugin jars are copied to `runelite/sideloaded-plugins/` and `~/.runelite/sideloaded-plugins/`.
- Runtime tuning lives in `rlbot-config.json`; document defaults when changing behavior.
- Do not commit secrets or user-specific paths; prefer env vars or config toggles.


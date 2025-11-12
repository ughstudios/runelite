# Repository Guidelines

## Project Structure & Module Organization
- Maven multi-module root: `cache/`, `runelite-api/`, `runelite-client/`, `runelite-jshell/`, `runelite-maven-plugin/`, plus standalone `rlbot-plugin/`.
- Each module keeps implementation under `src/main/java` and resources in `src/main/resources`; tests mirror under `src/test/java`.
- RuneLite upstream artifacts are expected in `runelite/runelite-client/target/`. Sideloaded plugin jars live in `runelite/sideloaded-plugins/` and `~/.runelite/sideloaded-plugins/`.

## Build, Test, and Development Commands
- `mvn -DskipTests clean install` — Build every module and publish jars to the local Maven repo.
- `mvn test` or `mvn -pl runelite-client test` — Execute the full test suite or focus on the client module when iterating quickly.
- `./rlbot-plugin/build-plugin.sh` — Compile and package the RLBot plugin only.
- `./hot-reload-plugin.sh` — Build the plugin and copy it to all sideload directories for rapid in-client reloads.
- `./run_runelite.sh` — Launch RuneLite; ensure `runelite/runelite-client/target/client-<ver>-shaded.jar` exists first.

## Coding Style & Naming Conventions
- Target Java 17 with 4-space indentation, adhering to repository `checkstyle.xml` and PMD rules under `runelite-client/pmd-ruleset.xml`.
- Packages live under `net.runelite.*`, with plugin logic in `net.runelite.client.plugins.rlbot`.
- Follow PascalCase for classes, camelCase for methods/fields, and UPPER_SNAKE_CASE for constants. Favor descriptive names that echo RuneScape terminology.

## Testing Guidelines
- Use JUnit 4 via Maven Surefire; name tests `*Test.java` and co-locate with the code under test.
- Prefer focused assertions that validate behavior over extensive mocking. Add regression tests when fixing defects.
- Run scope-specific tests (`mvn -pl runelite-client test`) before opening a PR, and execute any supporting scripts you introduce.

## Commit & Pull Request Guidelines
- Commit subjects stay imperative with optional module scope (e.g., `rlbot-plugin: add widget failover`).
- PRs must describe the change, rationale, linked issues, testing evidence, and screenshots/logs for UI or overlay adjustments. Call out configuration updates such as `rlbot-config.json` defaults.

## Security & Configuration Tips
- Do not commit secrets or developer-specific paths; lean on environment variables or documented config toggles.
- When changing runtime behavior, update defaults and notes in `rlbot-config.json`, and verify sideloaded plugin jars remain in the expected directories.

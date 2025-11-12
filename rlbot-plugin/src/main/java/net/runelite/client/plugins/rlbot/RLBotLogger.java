package net.runelite.client.plugins.rlbot;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import javax.inject.Inject;
import javax.inject.Singleton;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

// Logback (provided by RuneLite client at runtime)
import ch.qos.logback.classic.LoggerContext;
import ch.qos.logback.classic.encoder.PatternLayoutEncoder;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.rolling.RollingFileAppender;
import ch.qos.logback.core.rolling.SizeAndTimeBasedRollingPolicy;
import ch.qos.logback.core.util.FileSize;

/**
 * Handles logging for the RLBot plugin.
 * Now uses SLF4J with logback configuration for proper log rotation.
 */
@Singleton
public class RLBotLogger {
    
    private final RLBotConfig config;
    private final Logger logger;
    
    /**
     * Creates a new RLBotLogger with dependency injection.
     * 
     * @param config The plugin configuration
     */
    @Inject
    public RLBotLogger(RLBotConfig config) {
        this.config = config;
        this.logger = LoggerFactory.getLogger(RLBotLogger.class);
        try {
            // Ensure a dedicated rolling file appender is attached for the rlbot package
            setupFileAppender();
        } catch (Throwable t) {
            // Fall back silently if logback isn't available for some reason
            // Primary logging will still go to RuneLite's existing appenders
        }
        logger.info("RLBot logger initialized at: {}", LocalDateTime.now());
    }
    
    /**
     * Logs an info message.
     * 
     * @param message The message to log
     */
    public void info(String message) {
        RLBotConfig.LoggingLevel lvl = safeLevel();
        if (lvl == RLBotConfig.LoggingLevel.QUIET) {
            logger.debug(message);
        } else {
            logger.info(message);
        }
    }
    
    /**
     * Logs an error message.
     * 
     * @param message The message to log
     */
    public void error(String message) {
        logger.error(message);
    }
    
    /**
     * Logs a warning message.
     * 
     * @param message The message to log
     */
    public void warn(String message) {
        logger.warn(message);
    }
    
    /**
     * Logs a debug message if debug logging is enabled.
     * 
     * @param message The message to log
     */
    public void debug(String message) {
        RLBotConfig.LoggingLevel lvl = safeLevel();
        if (lvl == RLBotConfig.LoggingLevel.VERBOSE) {
            logger.debug(message);
        }
    }

    public void perf(String message) {
        RLBotConfig.LoggingLevel lvl = safeLevel();
        if (lvl == RLBotConfig.LoggingLevel.VERBOSE) {
            logger.info("[PERF] " + message);
        } else if (lvl == RLBotConfig.LoggingLevel.NORMAL) {
            logger.debug("[PERF] " + message);
        }
    }

    /**
     * Logs a message to file only (no console output).
     * Now uses debug level for file-only logging.
     * 
     * @param message The message to log
     */
    public void file(String message) {
        logger.debug("[FILE] {}", message);
    }
    
    /**
     * Logs an error message with an exception.
     * 
     * @param message The message to log
     * @param throwable The exception to log
     */
    public void error(String message, Throwable throwable) {
        logger.error(message, throwable);
    }

    private void setupFileAppender() throws Exception {
        // Attach a rolling file appender to the rlbot package logger so all plugin logs are captured
        String logFile = RLBotConstants.LOG_FILE;

        // Create parent directories if needed
        Path p = Paths.get(logFile).toAbsolutePath();
        Files.createDirectories(p.getParent());

        LoggerContext context = (LoggerContext) LoggerFactory.getILoggerFactory();

        // Use the package logger so classes under net.runelite.client.plugins.rlbot are captured
        ch.qos.logback.classic.Logger pkgLogger = context.getLogger("net.runelite.client.plugins.rlbot");

        final String appenderName = "RLBotPluginFile";
        if (pkgLogger.getAppender(appenderName) != null) {
            return; // already configured
        }

        PatternLayoutEncoder encoder = new PatternLayoutEncoder();
        encoder.setContext(context);
        encoder.setPattern("%d{yyyy-MM-dd HH:mm:ss.SSS} %-5level [%thread] %logger{0} - %msg%n");
        encoder.start();

        RollingFileAppender<ILoggingEvent> appender = new RollingFileAppender<>();
        appender.setContext(context);
        appender.setName(appenderName);
        appender.setFile(p.toString());

        SizeAndTimeBasedRollingPolicy<ILoggingEvent> policy = new SizeAndTimeBasedRollingPolicy<>();
        policy.setContext(context);
        // Place rotated logs next to the active log file
        File parent = p.getParent().toFile();
        String pattern = new File(parent, "rlbot-plugin_%d{yyyy-MM-dd}.%i.log").getAbsolutePath();
        policy.setFileNamePattern(pattern);
        policy.setMaxFileSize(FileSize.valueOf("10MB"));
        policy.setMaxHistory(7);
        policy.setTotalSizeCap(FileSize.valueOf("100MB"));
        policy.setParent(appender);
        policy.start();

        appender.setRollingPolicy(policy);
        appender.setEncoder(encoder);
        appender.start();

        pkgLogger.addAppender(appender);
        // Keep additivity so logs still flow to RuneLite's existing appenders
        pkgLogger.setAdditive(true);
    }

    private RLBotConfig.LoggingLevel safeLevel() {
        try {
            RLBotConfig.LoggingLevel lvl = config.logLevel();
            return lvl != null ? lvl : RLBotConfig.LoggingLevel.NORMAL;
        } catch (Exception e) {
            return RLBotConfig.LoggingLevel.NORMAL;
        }
    }
}

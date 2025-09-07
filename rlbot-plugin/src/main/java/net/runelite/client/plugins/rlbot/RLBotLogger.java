package net.runelite.client.plugins.rlbot;
import java.time.LocalDateTime;
import javax.inject.Singleton;
import javax.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
        logger.info("RLBot logger initialized at: {}", LocalDateTime.now());
    }
    
    /**
     * Logs an info message.
     * 
     * @param message The message to log
     */
    public void info(String message) {
        if (!config.quietLogging()) {
            logger.info(message);
        } else {
            logger.debug(message);
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
        if (config.debugLogging()) {
            logger.debug(message);
        }
    }

    public void perf(String message) {
        if (config.perfLogging()) {
            logger.info("[PERF] " + message);
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
} 
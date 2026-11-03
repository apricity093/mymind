package com.mymind.llm;

import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

@Component
public class SpringAiLlmGateway implements LlmGateway {

    private static final String STARTUP_PLACEHOLDER_KEY = "mymind-local-placeholder";

    private final ChatModel chatModel;
    private final String activeProfile;
    private final boolean modelCredentialsConfigured;

    public SpringAiLlmGateway(ObjectProvider<ChatModel> chatModelProvider,
                              @Value("${spring.profiles.active:anthropic}") String activeProfile,
                              @Value("${spring.ai.anthropic.api-key:}") String anthropicApiKey,
                              @Value("${spring.ai.deepseek.api-key:}") String deepSeekApiKey) {
        this.chatModel = chatModelProvider.getIfAvailable();
        this.activeProfile = activeProfile;
        this.modelCredentialsConfigured = credentialsConfigured(activeProfile, anthropicApiKey, deepSeekApiKey);
    }

    @Override
    public String chat(String systemPrompt, String userPrompt, double temperature, int maxTokens) {
        if (chatModel == null) {
            throw new IllegalStateException("Spring AI ChatModel is not configured");
        }
        if (!modelCredentialsConfigured) {
            throw new IllegalStateException("No real API key configured for active LLM profile: " + activeProfile);
        }
        List<Message> messages = new ArrayList<>();
        if (systemPrompt != null && !systemPrompt.isBlank()) {
            messages.add(new SystemMessage(clean(systemPrompt)));
        }
        messages.add(new UserMessage(clean(userPrompt)));
        ChatOptions options = ChatOptions.builder()
                .temperature(temperature)
                .maxTokens(maxTokens)
                .build();
        ChatResponse response = chatModel.call(new Prompt(messages, options));
        if (response == null || response.getResult() == null || response.getResult().getOutput() == null) {
            throw new IllegalStateException("Model returned no assistant message");
        }
        return response.getResult().getOutput().getText();
    }

    private static String clean(String value) {
        return value == null ? "" : value;
    }

    private static boolean credentialsConfigured(String activeProfile, String anthropicApiKey, String deepSeekApiKey) {
        String profile = clean(activeProfile).toLowerCase();
        if (profile.contains("deepseek")) {
            return hasRealKey(deepSeekApiKey);
        }
        return hasRealKey(anthropicApiKey);
    }

    private static boolean hasRealKey(String value) {
        String key = clean(value).trim();
        return !key.isBlank() && !STARTUP_PLACEHOLDER_KEY.equals(key);
    }
}

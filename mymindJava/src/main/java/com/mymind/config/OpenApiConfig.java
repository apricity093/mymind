package com.mymind.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.License;
import io.swagger.v3.oas.models.servers.Server;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
public class OpenApiConfig {

    @Bean
    public OpenAPI mymindOpenAPI() {
        return new OpenAPI()
                .info(new Info()
                        .title("MyMind Java Gateway API")
                        .version("0.1.0")
                        .description("Java 网关与多角色编排接口：知识工具熔断、应用监控和 Spring AI 模型调用。当前独立运行，未代理 Python Customer Policy 服务。")
                        .license(new License().name("Internal Project")))
                .servers(List.of(
                        new Server().url("http://localhost:8080").description("Local Java App"),
                        new Server().url("http://localhost:8081").description("Local Nginx Proxy")
                ));
    }
}

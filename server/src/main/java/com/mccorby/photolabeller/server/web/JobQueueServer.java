package com.mccorby.photolabeller.server.web;

import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;
import org.glassfish.jersey.media.multipart.MultiPartFeature;
import org.glassfish.jersey.server.ResourceConfig;
import org.glassfish.jersey.servlet.ServletContainer;


import org.eclipse.jetty.server.Request;
import org.eclipse.jetty.server.handler.AbstractHandler;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.ServletException;
import java.io.IOException;

public class JobQueueServer {
    // 예제 코드 가져와서 jetty는 정상인걸 확인.. 아마 화면 구성을 못해서 안뜨는게 아닐까 예상.
    // 일단 모델 만들어보자..
    public static void main(String[] args) {
        final ResourceConfig resourceConfig = new ResourceConfig(RestService.class);
        resourceConfig.register(MultiPartFeature.class);

        ServletHolder jerseyServlet = new ServletHolder(new ServletContainer(resourceConfig));

        Server jettyServer = new Server(9997);
        ServletContextHandler context = new ServletContextHandler(jettyServer, "/");
        context.addServlet(jerseyServlet, "/*");

        try {
            jettyServer.start();
            jettyServer.join();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            jettyServer.destroy();
        }
    }

}
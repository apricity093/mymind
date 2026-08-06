@echo off
set BASE_DIR=%~dp0
set "MAVEN_PROJECTBASEDIR=%BASE_DIR:~0,-1%"
set WRAPPER_JAR=%BASE_DIR%\.mvn\wrapper\maven-wrapper.jar
if not exist "%WRAPPER_JAR%" (
  powershell -Command "New-Item -ItemType Directory -Force '%BASE_DIR%\.mvn\wrapper' | Out-Null; Invoke-WebRequest -Uri 'https://repo.maven.apache.org/maven2/org/apache/maven/wrapper/maven-wrapper/3.3.2/maven-wrapper-3.3.2.jar' -OutFile '%WRAPPER_JAR%'"
)
if defined JAVA_HOME (
  set "JAVA_EXE=%JAVA_HOME%\bin\java.exe"
) else (
  set "JAVA_EXE=java"
)

"%JAVA_EXE%" -Dmaven.multiModuleProjectDirectory="%MAVEN_PROJECTBASEDIR%" -classpath "%WRAPPER_JAR%" org.apache.maven.wrapper.MavenWrapperMain %*

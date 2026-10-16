"""Point JAVA_HOME at a usable JDK (>= 11) before PyTerrier starts the JVM."""
import os

_CANDIDATES = [
    "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home",
    "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
    "/Library/Java/JavaVirtualMachines/amazon-corretto-11.jdk/Contents/Home",
    "/usr/lib/jvm/default-java",
    "/usr/lib/jvm/java-17-openjdk-amd64",
    "/usr/lib/jvm/java-11-openjdk-amd64",
]


def ensure_java() -> str | None:
    """Return the JAVA_HOME in use, setting it from known locations if unset."""
    if os.environ.get("JAVA_HOME"):
        return os.environ["JAVA_HOME"]
    for c in _CANDIDATES:
        if os.path.exists(os.path.join(c, "bin", "java")):
            os.environ["JAVA_HOME"] = c
            return c
    return None

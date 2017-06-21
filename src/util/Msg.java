package util;

public class Msg {

    private Msg() {
    }

    public static final void error(String msg, Object... args) {
        System.out.flush();
        System.err.println();
        System.err.printf(msg, args);
        System.err.println();
        System.err.flush();
        throw new RuntimeException();
    }

    public static final void warn(String msg, Object... args) {
        System.out.flush();
        System.err.println();
        System.err.println(String.format(msg, args));
        System.err.flush();
    }

}

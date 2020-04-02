package util;

public class Msg {

    private Msg() {
    }

    public static final void error(String msg, Object... args) {
        System.out.flush();
        System.out.println();
        System.out.printf(msg, args);
        System.out.println();
        System.out.flush();
        throw new RuntimeException();
    }

    public static final void warn(String msg, Object... args) {
        System.out.flush();
        System.out.println();
        System.out.println(String.format(msg, args));
        System.out.println();
        System.out.flush();
    }

}

package util;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Scanner;
import java.util.Set;

/**
 * Implements an option handler. An option is a flag-value pair of the form:
 *     -flag value
 * where flag is an alphanumeric string and value is of type integer, double, or String.
 *
 * Examples of String representations of option:
 *      - "-l 0.5"
 *      - "-S 2 -l 0.5"
 */
@SuppressWarnings("serial")
public class Options extends HashMap<String, String> {

    public Options() {
        super();
    }

    public Options(String opts) {
        super();
        parse(opts);
    }

    public void add(String opts) {
        parse(opts);
    }

    private String[] getFlags() {
        String[] flags = new String[size()];
        Set<String> keys = keySet();
        int i = 0;
        for (Iterator<String> it = keys.iterator(); it.hasNext(); ) {
            flags[i] = it.next();
            i++;
        }
        return flags;
    }

    public int getInt(String flag) {
        String val = get(flag);
        return (int) Double.parseDouble(val);
    }

    public double getDouble(String flag) {
        String val = get(flag);
        return Double.parseDouble(val);
    }

    public String getString(String flag) {
        return get(flag);
    }

    public String toString() {
        String[] keys = getFlags();
        StringBuilder sb = new StringBuilder();
        for (String k : keys) {
            sb.append(k).append(" ").append((get(k))).append(" ");
        }
        return sb.toString();
    }

    private void parse(String opts) {
        if (opts == null) {
            return;
        }
        Scanner sc = new Scanner(opts);
        while (sc.hasNext()) {
            String flag = sc.next();
            if ((flag.length() <= 1) || !flag.startsWith("-")) {
                exit("Invalid flag format for option: %score", flag);
            }
            if (sc.hasNext()) {
                String val = sc.next();
                put(flag, val);
            } else {
                exit("Missing value for option %score.", flag);
            }
        }
        sc.close();
    }

    private void exit(String msg, String flag) {
        System.err.printf(msg, flag);
        System.err.printf("%n  usage: {-flag value}*");
        System.exit(1);
    }
}

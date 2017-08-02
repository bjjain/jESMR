package util;

import java.io.FileWriter;
import java.io.IOException;

public class Writer {

    private final static String MSG = "Error! Writing to file failed.";

    private FileWriter writer;

    public static void append(String text, String file) {
        Writer w = new Writer();
        w.open(file, true);
        w.write(text);
        w.close();
    }

    private void open(String file, boolean append) {
        writer = null;
        try {
            writer = new FileWriter(file, true);
        } catch (IOException e) {
            System.err.println(MSG);
            e.printStackTrace();
        }
    }

    private void write(String str) {
        try {
            writer.write(str);
            writer.flush();
        } catch (IOException e) {
            System.err.println(MSG);
        }
    }


    private void close() {
        try {
            if (writer != null) {
                writer.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}

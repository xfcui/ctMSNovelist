package ch.moduled.fingerprintwrapper;

import java.io.IOException;
import java.io.OutputStream;

//Code from https://github.com/kpanic/openquake/blob/master/
public class PythonOutputStream extends OutputStream {
    private IPythonPipe thispipe;
    private StringBuilder buffer;

    public void setPythonStdout(IPythonPipe mypipe) {
        thispipe = mypipe;
        buffer = new StringBuilder();
    }

    @Override
    public void write(int arg0) throws IOException {
        buffer.append((char) arg0);
        if (arg0 == '\n') {
            thispipe.write(buffer.toString());
            buffer = new StringBuilder();
        }
    }
    
    @Override
    public void flush() {
    	thispipe.write(buffer.toString());
        buffer = new StringBuilder();
    }
}
import java.io.PrintStream;
import java.util.Random;

import com.cycling74.jitter.JitterMatrix;
import com.cycling74.max.*;
import com.cycling74.msp.*;

public class neural_fir extends MSPPerformer
{
	static class History {
		private final float[] _buffer;
		private int _pos = 0;
		
		public History(int size) {
			_buffer = new float[size];
		}
		
		public int size() {
			return _buffer.length;
		}
		
		public float get(int i) {
			return _buffer[(_pos + _buffer.length- i) % _buffer.length];
		}
		
		public void put(float v) {
			_pos = (_pos + 1) % _buffer.length;
			_buffer[_pos] = v;
		}
	}
	
	static class Matrix {
		private final float[] _data;
		private final int _M, _N;
		
		public Matrix(int M, int N) {
			_M = M;
			_N = N;
			_data = new float[M * N];
		}
		
		float get(int i, int j) {
			return _data[i * _N + j];
		}
		
		void set(int i, int j, float v) {
			_data[i * _N + j] = v;
		}
		
		int M() {
			return _M;
		}
		
		int N() {
			return _N;
		}

		public void clear() {
			neural_fir.clear(_data);
		}

		public float[] data() {
			return _data;
		}
	}
	
	private static final String[] INLET_ASSIST = new String[]{
		"input (sig)",
		"train (sig)"
	};
	private static final String[] OUTLET_ASSIST = new String[]{
		"output (sig)"
	};
	
	History _buffer;
	
	Matrix _hidden_coefs;
	float[] _hidden_bias;
	float[] _output_coefs;
	
	float[] _hidden_res;
	float[] _grad_by_hidden;
	
	Matrix _hidden_gradient;
	float[] _bias_gradient;
	float[] _output_gradient;
	
	JitterMatrix matrixOut;
	
	
	public float learn_rate = 0.001f;

	Random _rand = new Random();
	
	public void init() {
		learn_rate = orig_lr;
		for (int i = 0; i < _hidden_coefs.M(); ++i) {
			for (int j = 0; j < _hidden_coefs.N(); ++j) {
				_hidden_coefs.set(i, j, _rand.nextFloat() * 0.5f - 0.25f);
			}
			
		}
		for (int i = 0; i < _output_coefs.length; ++i) {
			_output_coefs[i] = _rand.nextFloat() - 0.5f;
			_hidden_bias[i] = _rand.nextFloat() * 0.1f + 0.01f;
		}
	}
	
	
	float orig_lr;
	public neural_fir(int buffer_size, int hidden_size, float learn_rate)
	{
		this.learn_rate = this.orig_lr = learn_rate;
		_buffer = new History(buffer_size);
		
		_hidden_coefs = new Matrix(buffer_size, hidden_size);
		_hidden_bias = new float[hidden_size];
		_output_coefs = new float[hidden_size];
		
		_hidden_res = new float[hidden_size];
		_grad_by_hidden = new float[hidden_size];
		
		_hidden_gradient = new Matrix(buffer_size, hidden_size);
		_bias_gradient = new float[hidden_size];
		_output_gradient = new float[hidden_size];
		
		init();
		
		declareInlets(new int[]{SIGNAL, SIGNAL});
		declareOutlets(new int[]{SIGNAL, DataTypes.FLOAT, DataTypes.MESSAGE});

		matrixOut = new JitterMatrix(1, "float", new int[]{buffer_size, hidden_size + 2});
		
		setInletAssist(INLET_ASSIST);
		setOutletAssist(OUTLET_ASSIST);
		this.declareAttribute("learn_rate");
	}
    
	
	
	public void dspsetup(MSPSignal[] ins, MSPSignal[] outs)
	{
		//If you forget the fields of MSPSignal you can select the classname above
		//and choose Open Class Reference For Selected Class.. from the Java menu
	}

	boolean _exc = false;
	
	public void perform(MSPSignal[] ins, MSPSignal[] outs)
	{
		if (_exc) {
			return;
		}
		try {
			float[] in = ins[0].vec;
			float[] out = outs[0].vec;
			if (ins[1].connected) {
				float[] ref = ins[1].vec;
				perform_learn(in, ref, out);
			} else {
				pure_perform(in, out);
			}
		} catch (Exception e) {
			e.printStackTrace(getPostStream());
			_exc = true;
		}
	}

	
	private void pure_perform(float[] in, float[] out) {
		for(int buf_pos = 0; buf_pos < in.length; buf_pos++)
		{
			_buffer.put(in[buf_pos]);
			out[buf_pos] = calculate_sample();
			
		}
	}
	
	private static void clear(float[] arr) {
		for (int i = 0; i < arr.length; ++i) {
			arr[i] = 0;
		}
	}

	private float calculate_sample() {
		float outv = 0;
		for (int j = 0; j < _hidden_res.length; ++j) {
			_hidden_res[j] = _hidden_bias[j];
			for (int i = 0; i < _buffer.size(); ++i) {
				_hidden_res[j] += _buffer.get(i) * _hidden_coefs.get(i, j);
			}
			if (_hidden_res[j] > 0) {
				outv += _hidden_res[j] * _output_coefs[j];
			}
		}
		return outv;
	}
	
	float err_sum;
	int it = 0;
	
	private void perform_learn(float[] in, float[] ref, float[] out) {
		
		
		if (it == 64) {
			update_weights();
			clear_gradients();
			it = 0;
			if (learn_rate > 0.1 * orig_lr) {
				learn_rate *= 0.99;
			}
			post(String.format("learn_rate = %f, error = %f", learn_rate, err_sum));
		}
		it++;
		err_sum = 0;
		for(int buf_pos = 0; buf_pos < in.length; buf_pos++)
		{
			_buffer.put(in[buf_pos]);
			float y = calculate_sample();
			out[buf_pos] = y;
			float error = (ref[buf_pos] - y);
			err_sum += error * error;
			compute_gradients(error);
		}
	}

	private void clear_gradients() {
		clear(_output_gradient);
		clear(_bias_gradient);
		_hidden_gradient.clear();
	}

	private void update_weights() {
		limitGrad(_output_gradient, 50);
		limitGrad(_bias_gradient, 50);
		limitGrad(_hidden_gradient.data(), 50);
		for (int j = 0; j < _output_coefs.length; ++j) {
			_output_coefs[j] += learn_rate * _output_gradient[j];
			_hidden_bias[j] += learn_rate * _bias_gradient[j];
			for (int i = 0; i < _buffer.size(); ++i) {
				_hidden_coefs.set(i, j, _hidden_coefs.get(i, j) + _hidden_gradient.get(i, j) * learn_rate);
			}
		}
	}

	private void compute_gradients(float error) {
		for (int j = 0; j < _output_coefs.length; ++j) {
			if (_hidden_res[j] > 0) {
				_output_gradient[j] += error * _hidden_res[j];
				float grad_by_hidden = error * _output_coefs[j];
				_bias_gradient[j] += grad_by_hidden;
				for (int i = 0; i < _buffer.size(); ++i) {
					_hidden_gradient.set(i, j, _hidden_gradient.get(i, j) + grad_by_hidden * _buffer.get(i));
				}
			}
		}
	}
	
	private static void limitGrad(float[] grad, float max) {
		float measure = 0;
		for (int i = 0; i < grad.length; ++i) {
			measure += grad[i] * grad[i];
		}
		if (measure > max) {
			for (int i = 0; i < grad.length; ++i) {
				grad[i] *= (max / measure);
			}
		}
	}
	
	public void bang() {
		for (int j = 0; j < _hidden_coefs._N; ++j) {
			matrixOut.setcell (new int[]{0, j}, new float[]{_hidden_bias[j] + 0.5f});
			for (int i = 0; i < _hidden_coefs._M; ++i) {
				matrixOut.setcell (new int[]{ i, j + 1}, new float[]{_hidden_coefs.get(i, j) + 0.5f});
			}
			matrixOut.setcell (new int[]{_hidden_coefs._M, j}, new float[]{_output_coefs[j] + 0.5f});

		}
		post(String.format("Err = %f", err_sum));
		this.outlet(1, err_sum);
		this.outlet(2, "jit_matrix", matrixOut.getName());
	}
}




using System;
using System.Collections;
using UnityEngine;
using System.Text;

/**
 * provides a quick and dirty performance counter
**/
public class FPSCounter : MonoBehaviour
{
	public TextMesh FPSCounterText;
	public TextMesh snapshotText;
	private StringBuilder strAccum = new StringBuilder(32);
	private float[] samples = new float[120];
	private int numSamples = 0;
	private int nextSampleIndex =0; //into ring buffer

	private float min, max, stdDev;
	public int updateEveryNFrames = 1;
	private int frameCounter = 0;

	public static Action<bool> RunTest;

	static float testSamples = 0;
	static int testFrames = 0;
	static bool runTest = false;


	void Awake()
	{
		gameObject.SetActive(false);
	}
	void OnRunTest(bool set)
	{
		if (set == runTest) return;
		Debug.Log($"FPSCounter OnRunTest {set}\n");
		StartCoroutine(_RunTest(set, 1.5f));
	}

	IEnumerator _RunTest(bool set, float delay)
	{
		yield return new WaitForSecondsRealtime(delay);
		runTest = set;

		if (set)
		{
			snapshotText.text = "Taking snapshot...";
			testFrames = 0;
			testSamples = 0;
		}
		else
		{
			if (testFrames != 0)
			{
				float ave = testFrames / testSamples;
				snapshotText.text = $"TEST OVER {testSamples}s\nDT: {1000f/ave}\nFPS: {ave}";
			}
		}


	}


	void Start()
	{
		snapshotText.text = "";
		RunTest += OnRunTest;
		if (FPSCounterText == null)
		{
			Debug.LogError("FPSCounter: no textbox, aborting.");
			enabled = false;
			return;
		}
	}

	void Update()
	{

		if (runTest)
		{
			testSamples += Time.unscaledDeltaTime;
			testFrames++;
		}

		float newSample = Time.unscaledDeltaTime;
		samples[nextSampleIndex] = newSample;
		numSamples = Mathf.Min(samples.Length, numSamples+1);
		nextSampleIndex = (nextSampleIndex+1)%samples.Length;

		frameCounter = (frameCounter+1)%updateEveryNFrames;
		if (frameCounter == 0)
		{
			float sum = 0.0f;
		//	max = samples[0];
		//	min = samples[0];
			for(int i=0; i<numSamples; i++)
			{
				sum += samples[i];
			//	max = Mathf.Max(samples[i], max);
		//		min = Mathf.Min(samples[i], min);
			}
			float ave = sum / (float)numSamples;
			/*
			float stdDevSum = 0.0f;
			if (numSamples > 0)
			{
				for(int i=0; i<numSamples; i++)
				{
					float diffMean = (samples[i] - ave);
					stdDevSum += diffMean*diffMean;
				}

				//note: using /N instead of /(N-1) for stddev
				stdDev = stdDevSum / (float)(numSamples-1);
				stdDev = Mathf.Sqrt(stdDev);
			}
			else
			{
				stdDev = -1.0f; //not valid
			}
			*/
			float aveFPS = 1.0f / ave;
			//float minFPS = 1.0f / min;
			//float maxFPS = 1.0f / max;
			strAccum.Clear();
			strAccum.Append("AVE:\n");
			strAccum.Append("\tDT: ").Append((ave*1000.0f).ToString("n2")).Append("ms\n");
			strAccum.Append("\tFPS: ").Append(aveFPS.ToString("n2")).Append("FPS\n");
			//strAccum.Append("StdDev: ").Append(stdDev.ToString("n3")).Append("ms\n");

			/*
			strAccum.Append("FASTEST: \n");
			strAccum.Append("\tDT: ").Append((min*1000.0f).ToString("n2")).Append("ms\n");
			strAccum.Append("\tFPS: ").Append((minFPS).ToString("n2")).Append("FPS\n");
			strAccum.Append("SLOWEST: \n");
			strAccum.Append("\tDT: ").Append((max*1000.0f).ToString("n2")).Append("ms\n");
			strAccum.Append("\tFPS: ").Append((maxFPS).ToString("n2")).Append("FPS\n");
			*/
			FPSCounterText.text = strAccum.ToString();
			strAccum.Clear();
		}
	}
}

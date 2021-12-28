using UnityEngine;

namespace KartGame.KartSystems
{
    /// <summary>
    /// A class which produces audio based on the speed that the kart is going.
    /// </summary>
    public partial class EngineAudio : MonoBehaviour
    {
        public ArcadeKart arcadeKart;

        [Range(0, 1)]
        public float RPM;
        [Space]
        [Tooltip("The minimum possible RPM of the engine.")]
        public float minRPM = 900;
        [Tooltip("The maximum possible RPM of the engine.")]
        public float maxRPM = 10000;
        [Space]
        [Tooltip("Increases randomness in engine audio.")]
        public float lumpyCamFactor = 0.05f;
        [Space]
        [Tooltip("Volume when at mininum RPM")]
        public float minVolume = 0.2f;
        [Tooltip("Volume when at maximum RPM")]
        public float maxVolume = 1.2f;
        [Space]
        [Tooltip("Smoothing of wave when a new stroke begins.")]
        public float strokeDamping = 0.1f;
        [Space]
        [Tooltip("Audio configuration for each engine stroke.")]
        public Stroke intake, compression, combustion, exhaust;
        [Tooltip("Map RPM to RPM^3")]
        public bool usePow = false;

        float m_NextStrokeTime;
        float m_Time;
        float m_SecondsPerSample;
        int m_Stroke;
        float[] m_RandomBuffer;
        float m_DeltaRPM;
        float m_LastRPM;
        float m_LastSampleL, m_LastSampleR;
        float m_Damper = 1f;
        float m_Volume = 1;

        void Awake()
        {
            m_RandomBuffer = new float[97];
            for (var i = 0; i < m_RandomBuffer.Length; i++)
                m_RandomBuffer[i] = Random.Range(-1, 1);
            intake.Init();
            compression.Init();
            combustion.Init();
            exhaust.Init();

            m_Stroke = 0;
            m_Time = 0;
            m_SecondsPerSample = 1f / AudioSettings.outputSampleRate;
        }

        void Update()
        {
            RPM = arcadeKart != null ? Mathf.Abs(arcadeKart.LocalSpeed())  : 0;
            m_DeltaRPM = RPM - m_LastRPM;

            //damp the movement of m_LastRPM
            m_LastRPM = Mathf.Lerp(m_LastRPM, RPM, Time.deltaTime * 100);
            if (Time.timeScale < 1)
                m_Volume = 0;
            else
                m_Volume = 1;
        }

        void OnAudioFilterRead(float[] data, int channels)
        {
            if (channels != 2)
                return;
            var r = usePow ? m_LastRPM * m_LastRPM * m_LastRPM : m_LastRPM;
            var gain = Mathf.Lerp(minVolume, maxVolume, r);

            //4 strokes per revolution
            var strokeDuration = 1f / ((Mathf.Lerp(minRPM, maxRPM, r) / 60f) * 2);

            for (var i = 0; i < data.Length; i += 2)
            {
                m_Time += m_SecondsPerSample;

                //a small random value use to mimic a "lumpy cam".
                var rnd = m_RandomBuffer[i % 97] * lumpyCamFactor;

                //is it time for the next stroke?
                if (m_Time >= m_NextStrokeTime)
                {
                    switch (m_Stroke)
                    {
                        case 0:
                            intake.Reset();
                            break;
                        case 1:
                            compression.Reset();
                            break;
                        case 2:
                            combustion.Reset();
                            break;
                        case 3:
                            exhaust.Reset();
                            break;
                    }

                    //increase the stroke counter
                    m_Stroke++;
                    if (m_Stroke >= 4) m_Stroke = 0;

                    //next stroke time has lump cam factor applied when rpm is decreasing (throttling down).
                    m_NextStrokeTime += strokeDuration + (strokeDuration * rnd * (m_DeltaRPM < 0 ? 1 : 0));

                    //damping resets every stroke, helps removes clicks and improves transition between strokes.
                    m_Damper = 0;
                }

                var sampleL = 0f;
                var sampleR = 0f;

                //In a 4 cylinder engine, all strokes would be playing simulataneously.
                switch (m_Stroke)
                {
                    case 0:
                        sampleL += intake.Sample() * rnd;
                        sampleR += compression.Sample();
                        sampleL += combustion.Sample();
                        sampleR += exhaust.Sample();
                        break;
                    case 1:
                        sampleR += intake.Sample();
                        sampleL += compression.Sample() * rnd;
                        sampleR += combustion.Sample();
                        sampleL += exhaust.Sample();
                        break;
                    case 2:
                        sampleR += intake.Sample();
                        sampleR += compression.Sample();
                        sampleL += combustion.Sample() * rnd;
                        sampleL += exhaust.Sample();
                        break;
                    case 3:
                        sampleL += intake.Sample();
                        sampleL += compression.Sample();
                        sampleR += combustion.Sample();
                        sampleR += exhaust.Sample() * rnd;
                        break;
                }

                m_Damper += strokeDamping;
                if (m_Damper > 1) m_Damper = 1;

                //smooth out samples between strokes
                sampleL = m_LastSampleL + (sampleL - m_LastSampleL) * m_Damper;
                sampleR = m_LastSampleR + (sampleR - m_LastSampleR) * m_Damper;
                sampleL = Mathf.Clamp(sampleL * gain, -1, 1);
                sampleR = Mathf.Clamp(sampleR * gain, -1, 1);
                data[i + 0] += sampleL + (sampleR * 0.75f);
                data[i + 0] *= m_Volume;
                data[i + 1] += sampleR + (sampleL * 0.75f);
                data[i + 1] *= m_Volume;
                m_LastSampleL = sampleL;
                m_LastSampleR = sampleR;
            }
        }
    }
}
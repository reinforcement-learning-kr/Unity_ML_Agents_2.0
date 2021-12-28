using UnityEngine;

namespace KartGame.KartSystems
{
    public partial class EngineAudio
    {
        /// <summary>
        /// Represents audio data for a single stroke of an engine (2 strokes per revolution)
        /// </summary>
        [System.Serializable] public struct Stroke
        {
            public AudioClip clip;
            [Range (0, 1)]
            public float gain;
            internal float[] buffer;
            internal int position;

            internal void Reset () => position = 0;

            internal float Sample ()
            {
                if (position < buffer.Length)
                {
                    var s = buffer[position];
                    position++;
                    return s * gain;
                }

                return 0;
            }

            internal void Init ()
            {
                //if no clip is available use a noisy sine wave as a place holder.
                //else initialise buffer of samples from clip data.
                if (clip == null)
                {
                    buffer = new float[4096];
                    for (var i = 0; i < buffer.Length; i++)
                        buffer[i] = Mathf.Sin (i * (1f / 44100) * 440) + Random.Range (-1, 1) * 0.05f;
                }
                else
                {
                    buffer = new float[clip.samples];
                    clip.GetData (buffer, 0);
                }
            }
        }
    }
}
using UnityEngine;

namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// A SensorComponent that creates a <see cref="VectorSensor"/>.
    /// </summary>
    [AddComponentMenu("ML Agents/Vector Sensor", (int)MenuGroup.Sensors)]
    public class VectorSensorComponent : SensorComponent
    {
        /// <summary>
        /// Name of the generated <see cref="VectorSensor"/> object.
        /// Note that changing this at runtime does not affect how the Agent sorts the sensors.
        /// </summary>
        public string SensorName
        {
            get { return m_SensorName; }
            set { m_SensorName = value; }
        }
        [HideInInspector, SerializeField]
        private string m_SensorName = "VectorSensor";

        /// <summary>
        /// The number of float observations in the VectorSensor
        /// </summary>
        public int ObservationSize
        {
            get { return m_ObservationSize; }
            set { m_ObservationSize = value; }
        }

        [HideInInspector, SerializeField]
        int m_ObservationSize;

        [HideInInspector, SerializeField]
        ObservationType m_ObservationType;

        VectorSensor m_Sensor;

        /// <summary>
        /// The type of the observation.
        /// </summary>
        public ObservationType ObservationType
        {
            get { return m_ObservationType; }
            set { m_ObservationType = value; }
        }

        /// <summary>
        /// Creates a VectorSensor.
        /// </summary>
        /// <returns></returns>
        public override ISensor[] CreateSensors()
        {
            m_Sensor = new VectorSensor(m_ObservationSize, m_SensorName, m_ObservationType);
            return new ISensor[] { m_Sensor };
        }

        /// <summary>
        /// Returns the underlying VectorSensor
        /// </summary>
        /// <returns></returns>
        public VectorSensor GetSensor()
        {
            return m_Sensor;
        }
    }
}

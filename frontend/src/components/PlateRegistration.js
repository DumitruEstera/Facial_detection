import React from 'react';

const PlateRegistration = ({ onRegister }) => {
  const [formData, setFormData] = React.useState({
    plate_number: '',
    vehicle_type: 'car',
    owner_name: '',
    owner_id: '',
    is_authorized: true,
    notes: ''
  });
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [message, setMessage] = React.useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setMessage('');

    try {
      const result = await onRegister(formData);
      setMessage({ type: 'success', text: result.message || 'License plate registered successfully!' });
      setFormData({
        plate_number: '',
        vehicle_type: 'car',
        owner_name: '',
        owner_id: '',
        is_authorized: true,
        notes: ''
      });
    } catch (error) {
      setMessage({ type: 'error', text: 'Error registering plate: ' + error.message });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  return (
    <div className="registration-form">
      <h2>🚗 Register License Plate</h2>
      
      {message && (
        <div className={`message ${message.type}`}>
          {message.text}
        </div>
      )}
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="plate_number">License Plate Number *</label>
          <input
            type="text"
            id="plate_number"
            name="plate_number"
            value={formData.plate_number}
            onChange={handleInputChange}
            required
            placeholder="e.g., B 123 ABC"
            style={{ textTransform: 'uppercase' }}
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="vehicle_type">Vehicle Type</label>
          <select
            id="vehicle_type"
            name="vehicle_type"
            value={formData.vehicle_type}
            onChange={handleInputChange}
          >
            <option value="car">Car</option>
            <option value="motorcycle">Motorcycle</option>
            <option value="truck">Truck</option>
            <option value="bus">Bus</option>
            <option value="van">Van</option>
          </select>
        </div>
        
        <div className="form-group">
          <label htmlFor="owner_name">Owner Name</label>
          <input
            type="text"
            id="owner_name"
            name="owner_name"
            value={formData.owner_name}
            onChange={handleInputChange}
            placeholder="Enter owner name"
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="owner_id">Owner ID</label>
          <input
            type="text"
            id="owner_id"
            name="owner_id"
            value={formData.owner_id}
            onChange={handleInputChange}
            placeholder="Enter owner/employee ID"
          />
        </div>
        
        <div className="form-group checkbox-group">
          <label>
            <input
              type="checkbox"
              name="is_authorized"
              checked={formData.is_authorized}
              onChange={handleInputChange}
            />
            <span>Authorized Vehicle</span>
          </label>
        </div>
        
        <div className="form-group">
          <label htmlFor="notes">Notes</label>
          <textarea
            id="notes"
            name="notes"
            value={formData.notes}
            onChange={handleInputChange}
            placeholder="Additional notes (optional)"
            rows="3"
          />
        </div>
        
        <button 
          type="submit" 
          disabled={isSubmitting || !formData.plate_number}
          className="btn btn-primary"
        >
          {isSubmitting ? '⏳ Registering...' : '✅ Register Plate'}
        </button>
      </form>
    </div>
  );
};

export default PlateRegistration;
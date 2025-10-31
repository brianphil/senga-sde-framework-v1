// src/components/OrderCreationPanel.tsx
import { useState } from "react";
import { Plus, Loader } from "lucide-react";
import { api } from "../services/api";
import { OrderCreateRequest, OrderResponse } from "../types";

// Kenyan destinations for testing
const KENYAN_LOCATIONS = [
  { name: "Nairobi CBD", lat: -1.2864, lon: 36.8172 },
  { name: "Westlands", lat: -1.2676, lon: 36.807 },
  { name: "Mombasa", lat: -4.0435, lon: 39.6682 },
  { name: "Kisumu", lat: -0.0917, lon: 34.768 },
  { name: "Nakuru", lat: -0.3031, lon: 36.08 },
  { name: "Eldoret", lat: 0.5143, lon: 35.2698 },
  { name: "Thika", lat: -1.0332, lon: 37.069 },
  { name: "Nyeri", lat: -0.4167, lon: 36.95 },
  { name: "Machakos", lat: -1.5177, lon: 37.2634 },
  { name: "Meru", lat: 0.05, lon: 37.65 },
];

interface Props {
  onOrderCreated: (order: OrderResponse) => void;
}

export default function OrderCreationPanel({ onOrderCreated }: Props) {
  const [creating, setCreating] = useState(false);
  const [formData, setFormData] = useState({
    customerName: "",
    customerPhone: "",
    pickupLocation: "Nairobi CBD",
    deliveryLocation: "Mombasa",
    packageWeight: 10,
    volumeM3: 0.5,
    priority: "standard" as "standard" | "urgent" | "emergency",
  });

  const handleCreate = async () => {
    if (!formData.customerName) {
      alert("Please enter customer name");
      return;
    }

    setCreating(true);
    try {
      const pickup = KENYAN_LOCATIONS.find(
        (l) => l.name === formData.pickupLocation
      )!;
      const delivery = KENYAN_LOCATIONS.find(
        (l) => l.name === formData.deliveryLocation
      )!;

      const orderRequest: OrderCreateRequest = {
        customer_name: formData.customerName,
        customer_phone: formData.customerPhone || undefined,
        pickup_location: {
          address: pickup.name,
          latitude: pickup.lat,
          longitude: pickup.lon,
        },
        delivery_location: {
          address: delivery.name,
          latitude: delivery.lat,
          longitude: delivery.lon,
        },
        package_weight: formData.packageWeight,
        volume_m3: formData.volumeM3,
        priority: formData.priority,
      };

      const order = await api.createOrder(orderRequest);
      onOrderCreated(order);

      // Reset form
      setFormData({
        ...formData,
        customerName: "",
        customerPhone: "",
      });
    } catch (error) {
      alert(`Failed to create order: ${error}`);
    } finally {
      setCreating(false);
    }
  };

  const handleQuickOrder = async () => {
    const names = [
      "John Mwangi",
      "Sarah Njeri",
      "David Ochieng",
      "Grace Wanjiru",
      "Peter Kamau",
    ];
    const randomName = names[Math.floor(Math.random() * names.length)];

    const pickups = KENYAN_LOCATIONS.slice(0, 2);
    const deliveries = KENYAN_LOCATIONS.slice(3, 5);

    const pickup = pickups[Math.floor(Math.random() * pickups.length)];
    const delivery = deliveries[Math.floor(Math.random() * deliveries.length)];

    setCreating(true);
    try {
      const orderRequest: OrderCreateRequest = {
        customer_name: randomName,
        pickup_location: {
          address: pickup.name,
          latitude: pickup.lat,
          longitude: pickup.lon,
        },
        delivery_location: {
          address: delivery.name,
          latitude: delivery.lat,
          longitude: delivery.lon,
        },
        package_weight: 500 + Math.random() * 150,
        volume_m3: 0.3 + Math.random() * 0.7,
        priority: Math.random() > 0.8 ? "urgent" : "standard",
      };

      const order = await api.createOrder(orderRequest);
      onOrderCreated(order);
    } catch (error) {
      alert(`Failed to create order: ${error}`);
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="panel">
      <h2 className="panel-title">Create Test Order</h2>

      <div className="form-group">
        <label>Customer Name *</label>
        <input
          type="text"
          value={formData.customerName}
          onChange={(e) =>
            setFormData({ ...formData, customerName: e.target.value })
          }
          placeholder="John Mwangi"
          disabled={creating}
        />
      </div>

      <div className="form-group">
        <label>Phone Number</label>
        <input
          type="text"
          value={formData.customerPhone}
          onChange={(e) =>
            setFormData({ ...formData, customerPhone: e.target.value })
          }
          placeholder="+254712345678"
          disabled={creating}
        />
      </div>

      <div className="form-row">
        <div className="form-group">
          <label>Pickup Location *</label>
          <select
            value={formData.pickupLocation}
            onChange={(e) =>
              setFormData({ ...formData, pickupLocation: e.target.value })
            }
            disabled={creating}
          >
            {KENYAN_LOCATIONS.map((loc) => (
              <option key={loc.name} value={loc.name}>
                {loc.name}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>Delivery Location *</label>
          <select
            value={formData.deliveryLocation}
            onChange={(e) =>
              setFormData({ ...formData, deliveryLocation: e.target.value })
            }
            disabled={creating}
          >
            {KENYAN_LOCATIONS.map((loc) => (
              <option key={loc.name} value={loc.name}>
                {loc.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="form-row">
        <div className="form-group">
          <label>Weight (kg)</label>
          <input
            type="number"
            value={formData.packageWeight}
            onChange={(e) =>
              setFormData({
                ...formData,
                packageWeight: parseFloat(e.target.value),
              })
            }
            min="0.1"
            step="0.1"
            disabled={creating}
          />
        </div>

        <div className="form-group">
          <label>Volume (m³)</label>
          <input
            type="number"
            value={formData.volumeM3}
            onChange={(e) =>
              setFormData({ ...formData, volumeM3: parseFloat(e.target.value) })
            }
            min="0.01"
            step="0.01"
            disabled={creating}
          />
        </div>
      </div>

      <div className="form-group">
        <label>Priority</label>
        <select
          value={formData.priority}
          onChange={(e) =>
            setFormData({ ...formData, priority: e.target.value as any })
          }
          disabled={creating}
        >
          <option value="standard">Standard</option>
          <option value="urgent">Urgent</option>
          <option value="emergency">Emergency</option>
        </select>
      </div>

      <div className="button-group">
        <button
          onClick={handleCreate}
          disabled={creating || !formData.customerName}
          className="btn-primary"
        >
          {creating ? (
            <>
              <Loader size={16} className="spinning" /> Creating...
            </>
          ) : (
            <>
              <Plus size={16} /> Create Order
            </>
          )}
        </button>

        <button
          onClick={handleQuickOrder}
          disabled={creating}
          className="btn-secondary"
        >
          Quick Random Order
        </button>
      </div>
    </div>
  );
}

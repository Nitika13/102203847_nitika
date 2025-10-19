import React, {useEffect, useState} from "react";
import axios from "axios";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

export default function Analytics(){
  const [data, setData] = useState(null);

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/analytics/summary").then(res => {
      setData(res.data);
    }).catch(err => {
      console.error(err);
    });
  }, []);

  if (!data) return <div>Loading analytics...</div>;

  const catData = Object.entries(data.categories || {}).map(([k,v]) => ({name:k, value:v}));

  return (
    <div>
      <h2>Analytics</h2>
      <div style={{width: "100%", height: 300}}>
        <ResponsiveContainer>
          <BarChart data={catData}>
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{marginTop:20}}>
        <h4>Price stats</h4>
        <pre>{JSON.stringify(data.price, null, 2)}</pre>
      </div>
    </div>
  );
}

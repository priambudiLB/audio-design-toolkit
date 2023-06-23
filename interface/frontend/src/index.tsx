import React from "react"
import ReactDOM from "react-dom"
import { StreamlitProvider } from "streamlit-component-lib-react-hooks"
import MyComponent from "./MyComponent"

ReactDOM.render(
  <StreamlitProvider>
    <MyComponent />
  </StreamlitProvider>
  ,
  document.getElementById("root")
)

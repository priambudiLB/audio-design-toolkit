import React from "react"
import ReactDOM from "react-dom"
import MyComponent from "./MyComponent"
import config from './config.json';

console.log(config)
ReactDOM.render(<MyComponent />,document.getElementById("root"))

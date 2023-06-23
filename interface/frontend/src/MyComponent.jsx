import { Streamlit } from "streamlit-component-lib"
import { useRenderData } from "streamlit-component-lib-react-hooks"
import React, { useState, useEffect, useCallback } from "react"

const debounce = (func, timeout = 1000) => {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      func.apply(this, args);
    }, timeout);
  };
};

const streamlitSetComponentValue = debounce(Streamlit.setComponentValue, 1000)
/**
 * This is a React-based component template with functional component and hooks.
 */
const MyComponent = () => {
  // let id = "knob1"
  // let lowVal = 1
  // let highVal = 100
  // let value = 8
  // let size = "medium"
  // let type = "Oscar"
  // let label = true
  // let name = "Default"

  let renderData = useRenderData()
  let id = renderData.args["id"]
  let lowVal = renderData.args["lowVal"]
  let highVal = renderData.args["highVal"]
  let value = renderData.args["value"]
  let size = renderData.args["size"]
  let type = renderData.args["type"]
  let label = renderData.args["label"]
  let name = renderData.args["name"]

  const [knobInUse, setKnobInUse] = useState({
    id: "",
    initY: 0,
    currentKnob: {}
  })

  let currentValue;
  if (value > highVal) {
    currentValue = highVal;
  } else if (value < lowVal) {
    currentValue = lowVal;
  } else {
    currentValue = value;
  }
  const scaler = 100 / (highVal - lowVal);

  if (size === "xlarge") {
    size = 128;
  } else if (size === "large") {
    size = 85;
  } else if (size === "medium") {
    size = 50;
  } else if (size === "small") {
    size = 40;
  } else {
    size = 30;
  }

  let initialSum =
    Math.floor(((value - lowVal) * scaler) / 2) * size;

  let initialKnobY = `translateY(-${initialSum}px)`;

  const imgFile = `${type}/${type}_${size}.png`;

  const [state, setState] = useState({
    id,
    lowVal,
    highVal,
    currentValue,
    scaler,
    type,
    label,
    size,
    imgFile,
    knobY: initialKnobY,
    name
  })

  const theme = renderData.theme
  // const theme = {
  //   font: 'Serif',
  //   textColor: 'black'
  // }

  function resetKnobInUse() {
    setKnobInUse({ id: "", initY: 0, value: 0, currentKnob: null })
  }

  const handleMouseDown = e => {
    setKnobInUse({
      id: id,
      initY: e.pageY,
      value: currentValue, //storing the value
      currentKnob: state //storing the reference
    })
  }

  const handleTouchStart = e => {
    setKnobInUse({
      id: id,
      initY: e.pageY,
      value: currentValue, //storing the value
      currentKnob: state //storing the reference
    })
  }

  const handleMouseMove = useCallback(function (e) {
    if (knobInUse.id !== "") {
      const oldState = { ...state };
      //freeze mouse drag activity if user hits top or bottom of the page
      if (e.pageY <= 10 || e.pageY >= document.body.clientHeight - 10) {
        setKnobInUse({ id: "", initY: 0, currentKnob: null })
        return;
      } else {
        //calculate new knob value
        oldState.currentValue =
          Math.round((oldState.currentValue =
            knobInUse.value +
            ((knobInUse.initY - e.pageY) * 1.7) /
            knobInUse.currentKnob.scaler + Number.EPSILON) * 100) / 100

        //use max/min variables for easier reading
        let max = knobInUse.currentKnob.highVal,
          min = knobInUse.currentKnob.lowVal;

        //ensure the know value does not exceed max and/or minimum values
        if (oldState.currentValue > max) {
          oldState.currentValue = max;
        } else if (oldState.currentValue < min) {
          oldState.currentValue = min;
        }
      }

      //update label (if user wants labels)
      if (knobInUse.currentKnob.label !== false) {
        oldState.label = oldState.currentValue
      }

      //change the image position to match
      let sum =
        (Math.floor(
          ((oldState.currentValue - knobInUse.currentKnob.lowVal) *
            knobInUse.currentKnob.scaler) /
          2
        ) +
          0) *
        knobInUse.currentKnob.size;

      let newY = `translateY(-${sum}px)`;
      //access to the image goes: container div > image wrapper div > image tag
      oldState.knobY = newY
      setState(oldState)
      streamlitSetComponentValue(oldState.currentValue)
    }
  }, [state, knobInUse])

  useEffect(() => {
    document.body.addEventListener("mousemove", handleMouseMove);
    document.body.addEventListener("touchmove", handleMouseMove);
    document.body.addEventListener("mouseup", resetKnobInUse);
    document.body.addEventListener("touchend", resetKnobInUse);
    return () => {
      document.body.removeEventListener('mousemove', handleMouseMove);
      document.body.removeEventListener('touchmove', handleMouseMove);
      document.body.removeEventListener("mouseup", resetKnobInUse);
      document.body.removeEventListener("touchend", resetKnobInUse);
    }
  }, [handleMouseMove])

  return (
    <>
      <div id={state.id} style={{ width: state.size * 2.5, display: 'flex', flexDirection: 'column', alignItems: 'center', marginBottom: 16 }}>
        <div style={{ textAlign: 'center', width: '100%', margin: '0px auto', marginBottom: 24, fontSize: '18px', fontWeight: 'bold', fontFamily: theme.font, color: theme.textColor }}>{state.name}</div>
        <div style={{ overflow: 'hidden', height: '50px', userSelect: 'none' }}>
          <img
            alt={state.id}
            draggable={false}
            style={{ transform: state.knobY }}
            src={"https://raw.githubusercontent.com/ColinBD/JSAudioKnobs/master/docs/knobs/" + state.imgFile}
            onMouseDown={handleMouseDown}
            onTouchStart={handleTouchStart}
          />
        </div>
        <div style={{ display: 'flex', width: '70%' }}>
          <div style={{ textAlign: 'center', width: '100%', margin: '-10px auto', fontSize: '14px', fontFamily: theme.font, color: theme.textColor }}>{state.lowVal}</div>
          <div style={{ textAlign: 'center', width: '100%', margin: '10px auto', fontSize: '16px', fontFamily: theme.font, fontWeight: 'bold', color: theme.textColor }}>{state.currentValue}</div>
          <div style={{ textAlign: 'center', width: '100%', margin: '-10px auto', fontSize: '14px', fontFamily: theme.font, color: theme.textColor }}>{state.highVal}</div>
        </div>
      </div>
    </>
  )
}

export default MyComponent

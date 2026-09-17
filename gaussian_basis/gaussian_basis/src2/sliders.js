const ENUM_CODES = {
    LINK: 0,
    MOUSE_SELECTOR: 1,
    TEXEL_SIDE_LENGTH: 2,
    TEXEL_SIDE_LENGTH_SELECTOR: 3,
    DATA_TEXEL_DIMENSIONS3_D: 4,
    PRESET_COMPOUNDS_DROPDOWN: 5,
    PRESET_ATOMS: 6,
    MAX_NUMBER_OF_ITERATIONS: 7,
    SOLVE: 8,
    CLEAR: 9,
    SHELL_METHOD_TYPE: 10,
    SIZE_SCALE: 11,
    SHOW_DENSITY: 12,
    WHICH_ORBITAL_SLIDER_VAL: 13,
    NUMBER_OF_PARTICLES: 14,
    VISUALIZATION_CONTROLS_START: 15,
    VISUALIZATION_SELECT: 16,
    USE_PERSPECTIVE_PROJECTION: 17,
    BRIGHTNESS: 18,
    VOLUME_RENDER_SECTION_START: 19,
    USE_LINEAR: 20,
    ALPHA_BRIGHTNESS: 21,
    COLOR_BRIGHTNESS: 22,
    VOLUME_TEXEL_DIMENSIONS3_D: 23,
    APPLY_BLUR: 24,
    BLUR_SIZE: 25,
    VOLUME_RENDER_SECTION_END: 26,
    PLANAR_SLICES_SECTION_START: 27,
    PLANAR_NORM_COORD_OFFSETS: 28,
    PLANAR_SLICES_SECTION_END: 29,
    ARROWS3_D_LINE_SECTION_START: 30,
    ARROW_DIMENSIONS: 31,
    USE_CONES: 32,
    ARROWS3_D_LINE_SECTION_END: 33,
    VISUALIZATION_CONTROLS_END: 34,
    TAKE_SCREENSHOTS: 35,
    CANVAS_HOVER_DISPLAY: 36,
    DUMMY_VALUE: 37,
};

let gVecParams = {};
let gUserParams = {};
let gCheckboxXorLists = {};

function createScalarParameterSlider(
    controls, enumCode, sliderLabelName, type, spec) {
    let label = document.createElement("label");
    label.for = spec['id']
    // label.style = "color:white; font-family:Arial, Helvetica, sans-serif";
    label.textContent = `${sliderLabelName} = ${spec.value}`;
    label.id = `slider-label-${enumCode}`;
    controls.appendChild(label);
    let slider = document.createElement("input");
    slider.type = "range";
    slider.style ="width: 95%;"
    if (isOnMobile())
        slider.style ="width: 90%; justify-content: center; "
    for (let k of Object.keys(spec))
        slider[k] = spec[k];
    slider.value = spec.value;
    slider.id = `slider-${enumCode}`;
    controls.appendChild(document.createElement("br"));
    controls.appendChild(slider);
    controls.appendChild(document.createElement("br"));
    slider.style.touchAction = 'none';
    if (isOnMobile())
        controls.appendChild(document.createElement("br"));
    slider.addEventListener("input", e => {
        let valueF = Number.parseFloat(e.target.value);
        let valueI = Number.parseInt(e.target.value);
        if (type === "float") {
            label.textContent = `${sliderLabelName} = ${valueF}`
            Module.set_float_param(enumCode, valueF);
        } else if (type === "int") {
            label.textContent = `${sliderLabelName} = ${valueI}`
            Module.set_int_param(enumCode, valueI);
        }
    });
};

function createCheckbox(controls, enumCode, name, value, xorListName='') {
    let label = document.createElement("label");
    // label.for = spec['id']
    // label.style = "color:white; font-family:Arial, Helvetica, sans-serif";
    label.innerHTML = `${name}`
    let checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = `checkbox-${enumCode}`;
    if (xorListName !== '') {
        if (!(xorListName in gCheckboxXorLists))
            gCheckboxXorLists[xorListName] = [checkbox.id];
        else
            gCheckboxXorLists[xorListName].push(checkbox.id);
    }
    // slider.style ="width: 95%;"
    // checkbox.value = value;
    checkbox.checked = value;
    // controls.appendChild(document.createElement("br"));
    controls.appendChild(checkbox);
    controls.appendChild(label);
    controls.appendChild(document.createElement("br"));
    checkbox.addEventListener("input", e => {
        console.log(e.target.checked);
        Module.set_bool_param(enumCode, e.target.checked);
        if (e.target.checked === true && xorListName !== '') {
            for (let id_ of gCheckboxXorLists[xorListName]) {
                if (id_ !== checkbox.id) {
                    let enumCode2 = parseInt(id_.split('-')[1]);
                    Module.set_bool_param(enumCode2, false);
                    document.getElementById(id_).checked = false;
                }
            }
        }
    }
    );
}


function editBoolDisplay(enumCode, value) {
    document.getElementById(`checkbox-${enumCode}`).checked = value;
}

function addExpansionSigns(value, text) {
    if (value !== null) {
        return ((value === true)? `▾ `: `▸ `) + text;
    }
}

function editScalarParameterSliderDisplay(enumCode, sliderLabelName, value) {
    let slider = document.getElementById(`slider-${enumCode}`);
    let label = document.getElementById(`slider-label-${enumCode}`);
    slider.value = value;
    label.textContent 
       = addExpansionSigns(
            false, `${sliderLabelName} = ${value}`);
}

function editVectorParameterSliderDisplay(enumCode, sliderLabelName, index, value) {
    let slider = document.getElementById(`slider-${enumCode}-${index}`);
    let label = document.getElementById(`slider-label-${enumCode}`);
    slider.value = value;
    gVecParams[sliderLabelName][Number.parseInt(index)] = value;
    label.textContent 
        = addExpansionsSigns(
            label.hidden,
            `${sliderLabelName} = (${gVecParams[sliderLabelName]})`);
}

function createVectorParameterSliders(
    controls, enumCode, sliderLabelName, type, spec) {
    let label = document.createElement("label");
    // label.style = "color:white; font-family:Arial, Helvetica, sans-serif";
    {
        label.textContent 
            = addExpansionSigns(
                !isOnMobile(),
                `${sliderLabelName} = (${spec.value})`);
    }
    gVecParams[sliderLabelName] = spec.value;
    label.id = `slider-label-${enumCode}`;
    label.className = `drop-down-label`;
    controls.appendChild(label);
    controls.appendChild(document.createElement("br"));
    // 
    let subDiv = document.createElement("div");
    controls.appendChild(subDiv);
    let subControls = subDiv;
    {
        subDiv.hidden = isOnMobile();
        // let subControls = (isOnMobile())? subDiv: controls;
        label.addEventListener(
            "click", e => {
                subControls.hidden = !subControls.hidden;
                if (label.textContent.at(0) === '▾')
                    label.textContent 
                        = '▸' + label.textContent.substring(1);
                else
                    label.textContent 
                        = '▾' + label.textContent.substring(1);
            }
        );
    }
    // 
    for (let i = 0; i < spec.value.length; i++) {
        let slider = document.createElement("input");
        slider.type = "range";
        // slider.style ="width: 95%;"
        slider.className = 'vec-slider';
        for (let k of Object.keys(spec))
            slider[k] = spec[k][i];
        slider.value = spec.value[i];
        slider.id = `slider-${enumCode}-${i}`;
        subControls.appendChild(slider);
        subControls.appendChild(document.createElement("br"));
        slider.style.touchAction = 'none';
        slider.addEventListener("input", e => {
            let valueF = Number.parseFloat(e.target.value);
            let valueI = Number.parseInt(e.target.value);
            if (type === "Vec2" || 
                type === "Vec3" || type === "Vec4") {
                gVecParams[sliderLabelName][i] = valueF;
                label.textContent
                    = addExpansionSigns(
                        true,
                        `${sliderLabelName} = (${gVecParams[sliderLabelName]})`);
                Module.set_vec_param(
                    enumCode, spec.value.length, i, valueF);
            } else if (type === "IVec2" || 
                        type === "IVec3" || type === "IVec4") {
                gVecParams[sliderLabelName][i] = valueI;
                label.textContent
                    = addExpansionSigns(
                        true,
                        `${sliderLabelName} = (${gVecParams[sliderLabelName]})`);
                Module.set_ivec_param(
                    enumCode, spec.value.length, i, valueI);
            }
        });
    }
    controls.appendChild(subControls);
    /* let altDiv = document.createElement("div");
    controls.appendChild(altDiv);
    altDiv.hidden = true;
    altDiv.className = 'vec-slider';
    altDiv.style.display = 'inline-block';
    // altDiv.style.width = '3em';
    for (let i = 0; i < spec.value.length; i++) {
        let newInput = document.createElement("input");
        newInput.type = "text";
        newInput.value = `${gVecParams[sliderLabelName][i]}`;
        newInput.style.width = '3em';
        altDiv.appendChild(newInput);
        // if (i !== spec.value.length - 1)
        //     altDiv.style['grid-template-columns'] += ' fr';
    }
    label.addEventListener("click", () => {
        altDiv.hidden = !altDiv.hidden;
        console.log(`${label.id} clicked`);
    });*/
    if (isOnMobile())
        controls.appendChild(document.createElement("br"));
};

function createSelectionList(
    controls, enumCode, defaultVal, selectionBoxName, textOptions
) {
    let label = document.createElement("label");
    // label.style = "color:white; font-family:Arial, Helvetica, sans-serif";
    label.textContent = selectionBoxName;
    controls.appendChild(label);
    controls.appendChild(document.createElement("br"));
    let selector = document.createElement("select");
    selector.className = 'dropdown';
    for (let i = 0; i < textOptions.length; i++) {
        let option = document.createElement("option");
        option.value = i;
        option.textContent = textOptions[i];
        selector.add(option);
    }
    selector.value = defaultVal;
    selector.addEventListener("change", e =>
        Module.selection_set(
            enumCode, Number.parseInt(e.target.value))
    );
    controls.appendChild(selector);
    controls.appendChild(document.createElement("br"));
}

function createUploadImage(
    controls, enumCode, name, w_code, h_code
) {
    let label = document.createElement("label");
    // label.style = "color:white; font-family:Arial, Helvetica, sans-serif";
    label.textContent = name;
    controls.appendChild(label);
    controls.appendChild(document.createElement("br"));
    // im.id = `image-${enumCode}`;
    let uploadImage = document.createElement("input");
    uploadImage.type = "file";
    let im = document.createElement("img");
    im.hidden = true;
    let imCanvas = document.createElement("canvas");
    imCanvas.hidden = true;
    controls.appendChild(uploadImage);
    // controls.appendChild(document.createElement("br"));
    controls.appendChild(im);
    // controls.appendChild(document.createElement("br"));
    controls.appendChild(imCanvas);
    // controls.appendChild(document.createElement("br"));
    uploadImage.addEventListener(
        "change", () => {
            console.log("image uploaded");
            const reader = new FileReader();
            reader.onload = e => {
                im.src = e.target.result;
            }
            let loadImageToPotentialFunc = () => {
                let ctx = imCanvas.getContext("2d");
                let width = Module.get_int_param(ENUM_CODES[w_code]);
                let height = Module.get_int_param(ENUM_CODES[h_code]);
                let imW = im.width;
                let imH = im.height;
                imCanvas.setAttribute("width", width);
                imCanvas.setAttribute("height", height);
                let heightOffset = 0;
                let widthOffset = 0;
                if (imW/imH >= width/height) {
                    let ratio = (imW/imH)/(width/height);
                    widthOffset = parseInt(0.5*width*(1.0 - ratio));
                    ctx.drawImage(im, widthOffset, heightOffset,
                                width*(imW/imH)/(width/height), height);
                } else {
                    let ratio = (imH/imW)/(height/width);
                    heightOffset = parseInt(0.5*height*(1.0 - ratio));
                    ctx.drawImage(im, widthOffset, heightOffset,
                                width, (imH/imW)/(height/width)*height);
                }
                let data = ctx.getImageData(0, 0, width, height).data;
                Module.image_set(
                    enumCode, data, width, height);
            }
            let promiseFunc = () => {
                if (im.width === 0 && im.height === 0) {
                    let p = new Promise(() => setTimeout(promiseFunc, 10));
                    return Promise.resolve(p);
                } else {
                    loadImageToPotentialFunc();
                }
            }
            reader.onloadend = () => {
                let p = new Promise(() => setTimeout(promiseFunc, 10));
                Promise.resolve(p);
            }
            reader.readAsDataURL(uploadImage.files[0]);
        }
    );
}

function modifyUserSliders(enumCode, variableList) {
    if (!(`${enumCode}` in gUserParams))
        gUserParams[`${enumCode}`] = {}; 
    for (let c of variableList) {
        if (!( c in gUserParams[`${enumCode}`]))
            gUserParams[`${enumCode}`][c] = 1.0;
    }
    let userSliders 
        = document.getElementById(`user-sliders-${enumCode}`);
    userSliders.textContent = ``;
    for (let v of variableList) {
        let label = document.createElement("label");
        // label.style = "color:white; font-family:Arial, Helvetica, sans-serif";
        label.textContent = `${v} = ${gUserParams[`${enumCode}`][v]}`;
        userSliders.appendChild(label);
        let slider = document.createElement("input");
        slider.type = "range";
        slider.style = "width: 95%;"
        slider.min = "-5";
        slider.max = "5";
        slider.step = "0.01";
        slider.value = gUserParams[`${enumCode}`][v];
        slider.addEventListener("input", e => {
            let value = Number.parseFloat(e.target.value);
            label.textContent = `${v} = ${value}`;
            gUserParams[`${enumCode}`][v] = value;
            Module.set_user_float_param(enumCode, v, value);
        });
        userSliders.appendChild(document.createElement("br"));
        userSliders.appendChild(slider);
        userSliders.appendChild(document.createElement("br"));
    }
}

function createEntryBoxes(
    controls, enumCode, entryBoxName, count, subLabels
) {
    let label = document.createElement("label");
    // label.style = "color:white; font-family:Arial, Helvetica, sans-serif";
    label.textContent = entryBoxName;
    controls.appendChild(label);
    controls.appendChild(document.createElement("br"));
    let entryBoxes = [];
    for (let i = 0; i < count; i++) {
        let entryBox = document.createElement('input');
        entryBox.type = "text";
        entryBox.value = "";
        entryBox.id = `entry-box-${enumCode}-${i}`;
        entryBox.style = "width: 95%;";
        let label = document.createElement("label");
        // label.style = "color:white; font-family:Arial, Helvetica, sans-serif";
        label.textContent = `${subLabels[i]}`;
        if (count >= 2) {
            controls.appendChild(label);
            controls.appendChild(document.createElement("br"));
        }
        controls.appendChild(entryBox);
        controls.appendChild(document.createElement("br"));
        entryBoxes.push(entryBox);
        entryBox.addEventListener("input", e =>
            Module.set_string_param(enumCode, i, `${e.target.value}`)
        );
    }
    let userSlidersDiv = document.createElement("div");
    userSlidersDiv.id = `user-sliders-${enumCode}`
    controls.appendChild(userSlidersDiv);

}

function createButton(
    controls, enumCode, buttonName, style=''
) {
    let button = document.createElement("button");
    button.innerText = buttonName;
    if (style !== '')
        button.style = style;
    controls.appendChild(button);
    controls.appendChild(document.createElement("br"));
    button.addEventListener("click", e => Module.button_pressed(enumCode));
}

function createLabel(
    controls, enumCode, labelName, style=''
) {
    let label = document.createElement("label");
    if (style !== '')
        label.style = style;
    label.textContent = `${labelName}`;
    label.id = `label-${enumCode}`;
    label.className = 'top-label';
    controls.appendChild(label);
    controls.appendChild(document.createElement("br"));
}

function createKaTeXLabel(
    controls, enumCode, latexText, style=''
) {
    let label = document.createElement("label");
    if (style !== '')
        label.style = style;
    label.textContent = `${latexText}`;
    label.id = `label-${enumCode}`;
    label.className = 'top-label';
    try {
        katex.render(`\\KaTeX \\space \\text{rendering} \\space \\text{here}.`, 
            label, {
            throwOnError: true
        });
    } catch {

    }
    controls.appendChild(label);
    controls.appendChild(document.createElement("br"));
}

function editKaTeXLabel(
    enumCode, latexText
) {
    let label = document.getElementById(`label-${enumCode}`);
    try {
        katex.render(latexText, 
            label, {
            throwOnError: true
        });
    } catch {

    }
}

function editLabel(enumCode, textContent) {
    let idVal = `label-${enumCode}`;
    let label = document.getElementById(idVal);
    label.textContent = textContent;
}

function createLineDivider(controls) {
    let hr = document.createElement("hr");
    hr.style = "color:white;"
    controls.appendChild(hr);
}

function createSubDiv(controls, name, style) {
    let div = document.createElement("div");
    let hr = document.createElement("hr");
    hr.style = "color:white;"
    controls.appendChild(hr);
    let label = document.createElement("label");
    if (style !== '')
        label.style = style;
    label.textContent = `+ ${name}`;
    // label.id = `label-${enumCode}`;
    label.className = 'top-label';
    controls.appendChild(label);
    controls.appendChild(document.createElement("br"));
    controls.appendChild(div);
    div.hidden = true;
    label.addEventListener(
        "click", e => {
            div.hidden = !div.hidden;
            if (label.textContent.at(0) === '+')
                label.textContent 
                    = '-' + label.textContent.substring(1);
            else
                label.textContent 
                    = '+' + label.textContent.substring(1);
        });
    return div;
}

function createHoveringLabelOnCanvas(enumCode, labelContent) {
    let div = document.createElement("div");
    div.style = `position: absolute; z-index: 10;`;
    div.id = `hovering-canvas-label-${enumCode}`;
    let label = document.createElement("label");
    label.textContent = labelContent;
    div.appendChild(label);
    document.getElementById('inner-div2').prepend(div);
}


function editHoveringCanvasLabelTextContent(
    enumCode, textContent) {
    let idVal = `hovering-canvas-label-${enumCode}`;
    let label = document.getElementById(idVal);
    label.textContent = textContent;
}

function editHoveringCanvasVisibilityTopLeftOffset(
    enumCode, isVisible, xPerc, yPerc
) {
    let idVal = `hovering-canvas-label-${enumCode}`;
    let label = document.getElementById(idVal);
    label.style['left'] = `${xPerc}%`;
    label.style['top'] = `${yPerc}%`;
    label.style['visibility'] = (isVisible)? 'visible': 'hidden';
}

function createLinkedLabel(controls, enumCode, labelContent, href) {
    let label = document.createElement("a");
    label.href = href;
    label.textContent = `${labelContent}`;
    label.id = `label-${enumCode}`;
    label.className = 'link-label';
    controls.appendChild(label);
    controls.appendChild(document.createElement("br"));
}

let controls = document.getElementById('controls');
createLinkedLabel(controls, 0, "Source", "https://github.com/marl0ny/hf");
createSelectionList(controls, 1, 0, "Mouse usage", [ "Rotate only",  "Place atom"]);
createSelectionList(controls, 3, 0, "Grid discretization size", [ "64x64x64",  "128x128x128",  "256x256x256"]);
createSelectionList(controls, 5, 1, "Preset Compounds", [ "Hydrogen molecule",  "Water",  "Carbon Dioxide",  "Oxygen Molecule"]);
createSelectionList(controls, 6, 0, "Atom dropdown", [ "H",  "He",  "Li",  "Be",  "B",  "C",  "N",  "O",  "F",  "Ne",  "Na",  "Mg",  "Al",  "Si",  "P",  "S",  "Cl",  "Ar",  "K",  "Ca",  "Sc",  "Ti",  "V",  "Cr",  "Mn",  "Fe",  "Co",  "Ni",  "Cu",  "Zn",  "Ga",  "Ge",  "As",  "Se",  "Br",  "Kr"]);
createScalarParameterSlider(controls, 7, "Max # of SCF steps", "int", {'value': 20, 'min': 0, 'max': 30});
createButton(controls, 8, "Solve");
createButton(controls, 9, "Clear");
createSelectionList(controls, 10, 0, "Method type", [ "All shells closed",  "Unrestricted"]);
createScalarParameterSlider(controls, 11, "Zoom out level", "float", {'value': 1.0, 'min': 0.5, 'max': 10.0, 'step': 0.01});
createCheckbox(controls, 12, "Show total electron density", true);
createScalarParameterSlider(controls, 13, "Which orbital", "int", {'value': 3, 'min': 0, 'max': 20});
createScalarParameterSlider(controls, 14, "Particle count upon reset", "int", {'value': 65536, 'min': 8192, 'max': 1048576, 'step': 4096});
let subControls0 = createSubDiv(controls, "Visualization Controls", "");
createSelectionList(subControls0, 16, 0, "Visualization select", [ "Volume render",  "Three orthogonal planar slices"]);
createCheckbox(subControls0, 17, "Use perspective projection", true);
createScalarParameterSlider(subControls0, 18, "Overall scaling", "float", {'value': 0.06, 'min': 0.0, 'max': 0.5, 'step': 0.01});
let subControls1 = createSubDiv(subControls0, "Volume Render Controls", "");
createCheckbox(subControls1, 20, "Linear interpolation", false);
createScalarParameterSlider(subControls1, 21, "Alpha brightness", "float", {'value': 2.0, 'min': 0.0, 'max': 10.0, 'step': 0.01});
createScalarParameterSlider(subControls1, 22, "Color brightness", "float", {'value': 1.0, 'min': 0.0, 'max': 10.0, 'step': 0.01});
createVectorParameterSliders(subControls1, 23, "Volume dimensions", "IVec3", {'value': [128, 128, 192], 'min': [16, 16, 16], 'max': [512, 512, 512], 'step': [2, 2, 4]});
createCheckbox(subControls1, 24, "Enable bloom", true);
createScalarParameterSlider(subControls1, 25, "Bloominess", "int", {'value': 5, 'min': 0, 'max': 10});
let subControls2 = createSubDiv(subControls0, "Three Orthogonal Planar Slices Controls", "");
createVectorParameterSliders(subControls2, 28, "Planar slices offsets (in normalized coordinates) for xy, yz, xz", "Vec3", {'value': [0.5, 0.5, 0.5], 'min': [0.0, 0.0, 0.0], 'max': [1.0, 1.0, 1.0], 'step': [0.001, 0.001, 0.001]});
let subControls3 = createSubDiv(subControls0, "Arrows Plot", "");
createVectorParameterSliders(subControls3, 31, "Arrows dimensions", "IVec3", {'value': [8, 8, 8], 'min': [8, 8, 8], 'max': [128, 128, 128]});
createCheckbox(subControls3, 32, "Use conical arrows", false);
createHoveringLabelOnCanvas(36, "");

// regions buttons
const regions = [
    "Cherkasy", "Chernihiv", "Chernivtsi", "Dnipro", 
    "Frankivsk", "Kharkiv", "Kherson", "Khmelnytskyi", "Kropivnitskiy", 
    "Kyiv", "Lviv", "Mykolaiv", "Odesa", "Poltava", "Rivne", "Sumy", 
    "Ternopil", "Vinnytsia", "Lutsk", "Uzhgorod", "Zaporizhzhia", "Zhytomyr"
];

const buttonContainer = document.querySelector('.button-container');

regions.forEach(region => {
    const button = document.createElement('button');
    button.classList.add('button_region');
    button.setAttribute('type', 'button');
    button.setAttribute('aria-label', 'toggle button');
    button.textContent = region;
    button.onclick = function() {
        toggleActive(this);
    };
    buttonContainer.appendChild(button);
});

function toggleActive(button) {
    var activeButtons = document.querySelectorAll('.button_region.active');
    if (activeButtons.length > 0) {
        activeButtons.forEach(function(activeButton) {
            activeButton.classList.remove('active');
        });
    }
    button.classList.toggle('active');


    // Отримайте обраний регіон з тексту кнопки
    var selectedRegion = button.textContent;

    // Оновіть вміст елемента для виведення вибраного регіону
    document.getElementById('selectedRegion').textContent = 'Selected region: ' + selectedRegion;
}

// slider
document.getElementById('timeRange').addEventListener('input', function() {
    var currentHour = new Date().getHours();
    var selectedHours = parseInt(this.value);

    if (currentHour + selectedHours >= 24) {
        currentHour = (currentHour + selectedHours) % 24;
    } else {
        currentHour += selectedHours;
    }
    var selectedTime = formatHour(currentHour) + ":00";
    document.getElementById('selectedHour').textContent = 'Selected hour: ' + selectedTime;
});

function formatHour(hour) {
    return hour < 10 ? "0" + hour : hour;
}

const slider = document.querySelector('.slider');
const selectedTime = document.getElementById('selectedTime');

slider.addEventListener('input', function() {
    selectedTime.textContent = this.value;
    this.style.setProperty('--value', (this.value - this.min) / (this.max - this.min));
});

// clock 
var hoursContainer = document.querySelector('.hours');
var minutesContainer = document.querySelector('.minutes');

var last = new Date(0);
last.setUTCHours(-1);

function updateTime() {
  var now = new Date;
  
  var lastHours = last.getHours().toString();
  var nowHours = now.getHours().toString();
  if (lastHours !== nowHours) {
    updateContainer(hoursContainer, nowHours);
  }
  
  var lastMinutes = last.getMinutes().toString();
  var nowMinutes = now.getMinutes().toString();
  if (lastMinutes !== nowMinutes) {
    updateContainer(minutesContainer, nowMinutes);
  }
  
  last = now;
}

function updateContainer(container, newTime) {
  var time = newTime.split('');
  
  if (time.length === 1) {
    time.unshift('0');
  }
  
  var first = container.firstElementChild;
  if (first.lastElementChild.textContent !== time[0]) {
    updateNumber(first, time[0]);
  }
  
  var last = container.lastElementChild;
  if (last.lastElementChild.textContent !== time[1]) {
    updateNumber(last, time[1]);
  }
}

function updateNumber(element, number) {
  element.lastElementChild.textContent = number;
}

setInterval(updateTime, 100);

// navbar
const menu = document.querySelector('.menu');
const btn = menu.querySelector('.nav-tgl');
btn.addEventListener('click', evt => {
	menu.classList.toggle('active');
})


// function getData() {
//     var selectedRegion = document.querySelector('.button_region.active').textContent;

//     fetch('http://13.48.5.18:8000/prediction', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json'
//         },
//         body: JSON.stringify({
//             region: selectedRegion,
//         })
//     })
//     .then(response => response.json())
//     .then(data => {
//         document.getElementById('result').innerText = JSON.stringify(data);
//     })
//     .catch(error => {
//         console.error('Error:', error);
//     });
// }

function getData() {
    var selectedRegion = document.querySelector('.button_region.active').textContent;

    fetch('http://13.48.5.18:8000/prediction', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            region: selectedRegion,
        })
    })
    .then(response => response.json())
    .then(data => {

        var currentHour = new Date().getHours();
        var selectedHour = parseInt(document.getElementById('timeRange').value);
        var index = (currentHour + selectedHour) % 24;
        var alarmProbability = data[index].float[1];
        // document.getElementById('proba').innerText = alarmProbability;
        var probabilityPercentage = (alarmProbability * 100).toFixed(2) + '%';
        document.getElementById('proba').innerText = probabilityPercentage;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}



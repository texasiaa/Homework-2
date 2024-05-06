// typed text
var typed = new Typed('#typed', 

{
    stringsElement: '#typed-strings',
    typeSpeed: 50,
    startDelay: 1000, 
    backSpeed: 20,
    loop: true,
    loopCount: Infinity,
    showCursor: true, 
    cursorChar: '|',
});

// animate heading
const swiftUpElements = document.querySelectorAll('.heading');

swiftUpElements.forEach(elem => {

	const words = elem.textContent.split(' ');
	elem.innerHTML = '';

	words.forEach((el, index) => {
		words[index] = `<span><i>${words[index]}</i></span>`;
	});

	elem.innerHTML = words.join(' ');

	const children = document.querySelectorAll('span > i');
	children.forEach((node, index) => {
		node.style.animationDelay = `${index * .2}s`;
	});

});

document.querySelectorAll('.btn').forEach(btn => {
  btn.addEventListener('click', () => {
    console.log(`Navigating to ${btn.textContent}`);
  });
});

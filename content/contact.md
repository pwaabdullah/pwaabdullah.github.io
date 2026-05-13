---
#title: "Contact"
url: "/contact/"
---



<div style="display: flex; justify-content: center; align-items: center; padding: 2rem 1rem;">
  <div style="
    backdrop-filter: blur(12px);
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    border-radius: 20px;
    padding: 2rem;
    max-width: 600px;
    width: 100%;
    color: white;
  ">

## Send Message to Abdullah

  <form id="contact-form" onsubmit="return handleValidatedSubmit(event)" style="display: flex; flex-direction: column; gap: 1rem;">
    <label>
      <span style="display: block; margin-bottom: 0.5rem;">Name</span>
      <input type="text" name="name" required style="
        width: 100%;
        padding: 0.75rem;
        border-radius: 10px;
        border: 1px solid #555;
        background-color: rgba(255,255,255,0.08);
        color: white;
        font-size: 1rem;
        outline: none;
      " />
    </label>

   <div>
    <label>
      <span style="display: block; margin-bottom: 0.5rem;">Email</span>
      <input type="email" name="email" required style="
        width: 100%;
        padding: 0.75rem;
        border-radius: 10px;
        border: 1px solid #555;
        background-color: rgba(255,255,255,0.08);
        color: white;
        font-size: 1rem;
        outline: none;
      " />
    </label>

   </div>
    <label>
      <span style="display: block; margin-bottom: 0.5rem;">Message</span>
      <textarea name="message" rows="5" required style="
        width: 100%;
        padding: 0.75rem;
        border-radius: 10px;
        border: 1px solid #555;
        background-color: rgba(255,255,255,0.08);
        color: white;
        font-size: 1rem;
        resize: vertical;
        outline: none;
      "></textarea>
    </label>

<div class="g-recaptcha" data-sitekey="6Le4LhwrAAAAADL-2U0w1gcQY7BiSrJTvgyvbPmO"></div>
   
<div>
    <button type="submit" style="
  margin-top: 1rem;
  padding: 0.75rem 1.5rem;
  background-color: #444;
  border: none;
  border-radius: 8px;
  color: white;
  font-size: 1rem;
  cursor: pointer;
  transition: background-color 0.3s ease;
" onmouseover="this.style.backgroundColor='#666';" onmouseout="this.style.backgroundColor='#444';">
  Send Message
</button>
   </div>
  </form>
  </div>
</div>

<div id="custom-alert" style="display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(0,0,0,0.85); color: white; padding: 1.5rem 2rem; border-radius: 10px; box-shadow: 0 8px 24px rgba(0,0,0,0.4); z-index: 1000; text-align: center;">
  ✅ Thanks! Your message has been sent.
  <div style="margin-top: 1rem;">
    <button onclick="document.getElementById('custom-alert').style.display='none'" style="background: transparent; border: 1px solid white; color: white; padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer;">Close</button>
  </div>
</div>

<!-- Social Media Buttons -->
<p align="center" style="padding-top: 40px; flex-wrap: wrap;">
  <a href="https://www.linkedin.com/comm/mynetwork/discovery-see-all?usecase=PEOPLE_FOLLOWS&followMember=newabdullah" target="_blank" class="social-btn linkedin">
    <i class="fab fa-linkedin"></i> Follow
  </a>
  <a href="https://github.com/pwaabdullah" target="_blank" class="social-btn github">
    <i class="fab fa-github"></i> GitHub
  </a>
  <a href="https://x.com/newmamun" target="_blank" class="social-btn x">
    <svg style="height: 20px; width: 20px; fill: white;" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
      <path d="M17.53 2H21l-7.22 8.24L23 22h-6.38l-4.99-6.03L5.8 22H2l7.66-8.75L1 2h6.5l4.44 5.37L17.53 2Zm-1.1 18h1.74L8.61 4H6.75l9.68 16Z"/>
    </svg>
  </a>



</p>

<style>

  .social-btn {
  display: inline-block;
  margin: 10px;
  padding: 12px 25px;
  font-size: 14px;
  color: white;
  text-decoration: none;
  border-radius: 30px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
  transition: all 0.3s ease-in-out;
  font-family: 'Segoe UI', sans-serif;
}
/* Responsive reCAPTCHA */


/* Mobile responsiveness (optional) */
@media (max-width: 768px) {
  .g-recaptcha {
    transform: scale(0.77); /* Scale down the widget */
    transform-origin: 0 0; /* Keep the widget aligned to the top-left corner */
  }
}

.social-btn i {
  margin-right: 10px;
}

.social-btn.linkedin {
  background-color: #0077B5;
}

.social-btn.github {
  background-color: #333;
}

.social-btn.x {
  background-color: #000000;
}



.social-btn:hover {
  transform: translateY(-3px);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}


@media screen and (max-width: 768px) {
  .bio-section {
    padding: 20px;
    font-size: 0.95rem;
  }

  .bio-section h2 {
    font-size: 1.3rem;
  }

  .social-btn {
    display: block;
    width: 60%;
    margin: 20px auto;
    text-align: center;
    font-size: 16px;
  }

  p img {
    width: 90% !important;
  }
}
</style>
<script>
  function validateRecaptcha() {
    const response = grecaptcha.getResponse();
    if (!response) {
      alert("Please complete the reCAPTCHA.");
      return false;
    }
    return true;
  }

  async function handleValidatedSubmit(event) {
    event.preventDefault();
    if (!validateRecaptcha()) {
      return false;
    }

    const form = event.target;
    const data = new FormData(form);
    const response = await fetch("https://formspree.io/f/moveoagg", {
      method: "POST",
      body: data,
      headers: { 'Accept': 'application/json' }
    });

    if (response.ok) {
      form.reset();
      const alertBox = document.getElementById('custom-alert');
      alertBox.style.display = 'block';
      setTimeout(() => {
        alertBox.style.display = 'none';
      }, 3000);
    } else {
      alert("Oops! Something went wrong.");
    }

    return false;
  }
</script>
<script src="https://www.google.com/recaptcha/api.js" async defer></script>
<!-- Font Awesome for Icons -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
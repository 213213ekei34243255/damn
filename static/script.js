document.addEventListener("DOMContentLoaded", function () {
  const _0xe845e2 = document.getElementById('contact-form');
  const _0x283810 = document.querySelector("#menu-icon");
  const _0x323837 = document.querySelector(".navbar");
  const _0x1b2aab = document.querySelectorAll("section");
  const _0x2f2a24 = document.querySelectorAll("header nav a");
  const _0x3f712d = document.querySelector("header");
  window.addEventListener('scroll', function () {
    const _0x49f664 = document.querySelector(".header");
    const _0x35380d = document.querySelector('#home');
    const _0x3722f1 = _0x35380d.offsetHeight;
    if (window.scrollY > _0x3722f1) {
      _0x49f664.classList.add("sticky");
    } else {
      _0x49f664.classList.remove('sticky');
    }
  });
  function _0x31d829() {
    const _0x320600 = _0xe845e2.querySelector("input[name=\"name\"]");
    const _0x563ae4 = _0xe845e2.querySelector("input[name=\"email\"]");
    const _0x4a0637 = _0xe845e2.querySelector("input[name=\"mobile\"]");
    const _0x2d8c79 = _0xe845e2.querySelector("input[name=\"subject\"]");
    const _0x6403ee = _0xe845e2.querySelector("textarea[name=\"message\"]");
    const _0x5b9543 = "Full Name: " + _0x320600.value + "<br> Email: " + _0x563ae4.value + "<br> Mobile Number: " + _0x4a0637.value + "<br> Subject: " + _0x2d8c79.value + "<br> Message: " + _0x6403ee.value;
    Email.send({
      'Host': 'smtp.elasticemail.com',
      'Port': 0x9dd,
      'Username': "abrahamshaunindustries@gmail.com",
      'Password': '8DF549C6DA82F41752402C7D068705E5B1B4',
      'To': "abrahamshaunindustries@gmail.com",
      'From': "abrahamshaunindustries@gmail.com",
      'Subject': _0x2d8c79.value,
      'Body': _0x5b9543
    }).then(_0x2769f4 => {
      console.log("Email sent successfully:", _0x2769f4);
      Swal.fire({
        'title': "Success!",
        'text': "Message sent successfully!",
        'icon': "success"
      });
    })["catch"](_0x3bdfd9 => {
      console.error("Error sending email:", _0x3bdfd9);
      Swal.fire({
        'title': "Error!",
        'text': "Error sending email. Please try again later.",
        'icon': "error"
      });
    });
  }
  _0x283810.onclick = () => {
    _0x283810.classList.toggle("bx-x");
    _0x323837.classList.toggle("active");
  };
  window.onscroll = () => {
    _0x1b2aab.forEach(_0x3c3200 => {
      let _0x4b057a = window.scrollY;
      let _0x103f15 = _0x3c3200.offsetTop - 0x96;
      let _0x47bb34 = _0x3c3200.offsetHeight;
      let _0x45fdf7 = _0x3c3200.getAttribute('id');
      if (_0x4b057a >= _0x103f15 && _0x4b057a < _0x103f15 + _0x47bb34) {
        _0x2f2a24.forEach(_0x3cae67 => {
          _0x3cae67.classList.remove("active");
        });
        document.querySelector("header nav a[href*=\"" + _0x45fdf7 + "\"]").classList.add("active");
      }
    });
    _0x3f712d.classList.toggle("sticky", window.scrollY > 0x64);
    _0x283810.classList.remove("bx-x");
    _0x323837.classList.remove('active');
  };
  _0xe845e2.addEventListener("submit", function (_0x4bd606) {
    _0x4bd606.preventDefault();
    _0x31d829();
  });
});
class Chatbox {
  constructor() {
    this.args = {
      'openButton': document.querySelector(".chatbox__button"),
      'chatBox': document.querySelector(".chatbox__support"),
      'sendButton': document.querySelector(".send__button"),
      'userInput': document.querySelector(".chatbox__support input")
    };
    this.state = false;
    this.messages = [];
    this.baseUrl = '';
  }
  ['display']() {
    const {
      openButton: _0x74a4aa,
      chatBox: _0x236c6f,
      sendButton: _0xf76495,
      userInput: _0xa1965
    } = this.args;
    _0x74a4aa.addEventListener("click", () => this.toggleState(_0x236c6f));
    _0xf76495.addEventListener("click", () => this.onSendButton(_0x236c6f));
    _0xa1965.addEventListener("keyup", ({
      key: _0x4376a8
    }) => {
      if (_0x4376a8 === "Enter") {
        this.onSendButton(_0x236c6f);
      }
    });
  }
  ['toggleState'](_0x5a019e) {
    this.state = !this.state;
    if (this.state) {
      _0x5a019e.classList.add("chatbox--active");
    } else {
      _0x5a019e.classList.remove("chatbox--active");
    }
  }
  ['onSendButton'](_0x44a14e) {
    const _0x4ce0b7 = this.args.userInput;
    const _0x9d4c7 = _0x4ce0b7.value.trim();
    if (_0x9d4c7 === '') {
      return;
    }
    let _0x451eca = {
      'name': "User",
      'message': _0x9d4c7
    };
    this.messages.push(_0x451eca);
    this.updateChatText(_0x44a14e);
    fetch(this.baseUrl + '/predict', {
      'method': "POST",
      body: JSON.stringify({

            mode: "agent",
        
            goal: _0x9d4c7,
        
            session_id: "web",
        
            observation: {
        
                url: window.location.href,
        
                title: document.title,
        
                text: document.body.innerText.substring(0, 4000)
        
            },
        
            memory: {}
        
        }),
      'headers': {
        'Content-Type': 'application/json'
      }
    }).then(_0x3adffb => _0x3adffb.json()).then(_0x305b1f => {
      console.log("Gemini API response:", _0x305b1f);
      let _0x585a6b = {
          name: "Noah",
          message: JSON.stringify(_0x305b1f, null, 2)
      };
      this.messages.push(_0x585a6b);
      this.updateChatText(_0x44a14e);
      if (_0x305b1f.url) {
        this.fetchFirstFiveSentences(_0x305b1f.url).then(_0x1abc4d => {
          let _0xef6df = {
            'name': "Veronica",
            'message': _0x1abc4d
          };
          this.messages.push(_0xef6df);
          this.updateChatText(_0x44a14e);
          let _0x2da353 = {
            'name': "Veronica",
            'message': "You can read more here: " + _0x305b1f.url
          };
          this.messages.push(_0x2da353);
          this.updateChatText(_0x44a14e);
        })["catch"](_0x16fe49 => console.error('Error:', _0x16fe49));
      }
    })["catch"](_0xc0d0b2 => {
      console.error("Error with Gemini fetch:", _0xc0d0b2);
    });
  }
  ['fetchFirstFiveSentences'](_0x21e1db) {
    return fetch(this.baseUrl + '/fetch_sentences', {
      'method': "POST",
      'body': JSON.stringify({
        'url': _0x21e1db
      }),
      'headers': {
        'Content-Type': "application/json"
      }
    }).then(_0x1347ba => _0x1347ba.json()).then(_0x1178f6 => _0x1178f6.first_five_sentences)['catch'](_0x520b79 => {
      console.error("Error:", _0x520b79);
      throw _0x520b79;
    });
  }
  ["promptForLearning"](_0x264aea) {
    const _0x23a6ef = this.args.userInput;
    const _0x490746 = "Can you please teach me the answer to this question or type 'skip' to skip?\n\n" + _0x264aea;
    let _0x5ecbf3 = {
      'name': "Veronica",
      'message': _0x490746
    };
    this.messages.push(_0x5ecbf3);
    this.updateChatText(this.args.chatBox);
    _0x23a6ef.disabled = false;
    _0x23a6ef.addEventListener('keyup', ({
      key: _0x1c8cf1
    }) => {
      if (_0x1c8cf1 === "Enter") {
        const _0x1e1462 = _0x23a6ef.value.trim();
        if (_0x1e1462.toLowerCase() === "skip") {
          this.clearMessages();
        } else {
          this.sendLearningData(_0x264aea, _0x1e1462);
        }
      }
    });
  }
  ['sendLearningData'](_0x48055a, _0x319f62) {
    const _0xcd6efc = this.args.userInput;
    fetch(this.baseUrl + "/learn", {
      'method': "POST",
      'body': JSON.stringify({
        'question': _0x48055a,
        'answer': _0x319f62
      }),
      'headers': {
        'Content-Type': 'application/json'
      }
    }).then(_0x27d4e1 => _0x27d4e1.json()).then(_0x21e352 => {
      console.log(_0x21e352.message);
      _0xcd6efc.value = '';
      _0xcd6efc.disabled = false;
      this.clearMessages();
    })['catch'](_0x25752c => {
      console.error("Error:", _0x25752c);
      _0xcd6efc.value = '';
      _0xcd6efc.disabled = false;
      this.clearMessages();
    });
  }
  ['clearMessages']() {
    const {
      chatBox: _0x2286d4
    } = this.args;
    _0x2286d4.querySelector(".chatbox__messages").innerHTML = '';
    this.messages = [];
  }
  ['updateChatText'](_0x1cf644) {
    var _0x360b28 = '';
    this.messages.slice().reverse().forEach(function (_0x336627, _0x4b681b) {
      if (_0x336627.name === "Veronica") {
        _0x360b28 += "<div class=\"messages__item messages__item--visitor\">" + _0x336627.message + "</div>";
      } else {
        _0x360b28 += "<div class=\"messages__item messages__item--operator\">" + _0x336627.message + '</div>';
      }
    });
    const _0x412ba5 = _0x1cf644.querySelector(".chatbox__messages");
    _0x412ba5.innerHTML = _0x360b28;
  }
}
const chatbox = new Chatbox();
chatbox.display();
document.addEventListener("DOMContentLoaded", function () {
  const _0x4cb736 = document.querySelectorAll("section");
  const _0x29cc6c = document.getElementById("chat");
  window.addEventListener("scroll", function () {
    let _0x59bbaa = false;
    _0x4cb736.forEach(_0x2164bd => {
      const _0x5a9844 = _0x2164bd.getBoundingClientRect();
      if (_0x5a9844.top <= window.innerHeight && _0x5a9844.bottom >= 0x0) {
        if (_0x2164bd === _0x29cc6c) {
          _0x59bbaa = true;
        }
      }
    });
    if (_0x59bbaa) {
      document.body.classList.add("chat-active");
    } else {
      document.body.classList.remove("chat-active");
    }
  });
});

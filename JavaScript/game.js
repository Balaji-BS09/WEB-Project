let userScore = 0;
let compScore = 0;

const choices = document.querySelectorAll(".choice");
const msg = document.querySelector("#msg");
const userScorePara = document.querySelector("#user-score");
const compScorePara = document.querySelector("#comp-score");

const genCompChoice = () => {
    const options = ["rock" , "paper", "scissor"];
    const randIdx = Math.floor(Math.random()*3);
    return options[randIdx];
}

const drawGame = () => {

    msg.innerText = "Game was draw!,Try Again.";
}
let userWin =  true;
const showWinner = () => {
    if(userWin) {
        userScore++;
        userScorePara.innerText = userScore;
        msg.innerText = "You Win!";
    }
    else {
        compScore++;
        compScorePara.innerText = compScore;
        msg.innerText = "You Lose!";
    }
}

const playGame = (userChoice) => {
   
    const  compChoice = genCompChoice();
  

    if(userChoice === compChoice) {
        drawGame();
    }else{
        
        if(userChoice === "rock") {
            userWin =compChoice === "paper" ? false : true;
        }else if(userChoice === "paper") {
            userWin = compChoice === "scissor" ? false : true;
        }else {
            userWin = compChoice === "rock" ? false : true;
        }
        showWinner(userWin);
    }

}

choices.forEach((choice) => {
    choice.addEventListener("click", () => {
        const userChoice = choice.getAttribute("id");
        playGame(userChoice);
    })
})
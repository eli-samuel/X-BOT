# X-BOT
## _Full Release Notes and Information_

## About
X-BOT is a Discord bot desgined to provide entertainment, as well as some utilities. This bot serves as a playground for us to implement our own functionalities to either give people a laugh or automate certain tasks.

## Main features
- Basic Anti-spam filter
- Automated good morning messages
- Customized responses to specific messages

## Features to add
- Host on a server to allow for runtime with little to no downtime
- Neural network with message predictors

## Patch notes

### *April 1, 2023*
`v1.3.2 release`
##### YOMNOMNOMNOM part 2

> Developer comments: YOMNOMNOM was too boring, updated the audio file.

    - !yom

---
### *March 31, 2023*
`v1.3.1 release`
##### YOMNOMNOMNOM

> Developer comments: Time to implement voice chatting. This feature was suggested by tokki and sacha.

    - !yom joins voice and plays an audio clip
    - "go away" forces X-Bot to leave the voice channel it is currently in

##### Funky Friday
> Developer comments: ???
    - ???
---
### *March 22, 2023*
`v1.2.5 release`
##### Period

> Developer comments: I am too lazy to write "period". This patch enables my laziness.

    - Responds "period" to any "." message
---
### *March 9, 2023*
`v1.2.5 release`
##### Added a good morning message

> Developer comments: This responds to people sending "good morning". This feature was suggested by bucket.

    - Responds "hi" to any "good morning" message
    
##### Modified the reactions to specific users

> Developer comments: Jocelyn has been *X* reacted for months now, it was time to swap.

    - Thumbs up for Jocelyn
    - X and thumbs down for Logan
    
---
### *February 8, 2023*
`v1.2.4 release`
##### Probability changes

> Developer comments: Some reactions were happening too often, so we gotta make them more rare.

    - F reaction probability lowered from 2% to 1%
    - heart reaction probability lowered from 2% to 1%
    - X reaction probability lowered from 2% to 1%
    
---
### *January 27, 2023*
`v1.2.3 release`
##### Added an 8-ball functionality

> Developer comments: Who doesn't like being told answers to any question? 
Okay maybe not any question but a yes or no question.

    - If any message is sent with the word "xbot" in it and ending with a "?" the user will receive a response
    - There are ten different responses all with equal probability of being sent

---
### *January 26, 2023*
`v1.2.2 bugfix`
##### Modified good morning GIFs

> Developer comments: There seemed to be an issue with the previous message sending. This patch attempts to fix it.

    - Changed the time of message sending to 9 AM EST
    - Fixed an error with the date and time not being properly calculated
    
---
### *January 20, 2023*
`v1.2.1 pre-release`
##### Added good morning GIFs

> Developer comments: Many users send good morning images/GIFs/messages, this will just be another one.

    - Send a day-specific GIF and message every morning

---
### *January 13, 2023*
`v1.1.3 release`
##### Added reactions

> Developer comments: People really liked getting **F** and **heart** reacted, this gives them more to like!

    - 2% chance to react with X

---
### *January 5, 2023*
`v1.1.2 release`
##### Changed the way the users interact with the *high* role

> Developer comments: It didn't really make sense to have people have the role permanently, so this update changes how the role is given/taken away.

    - Changed keyword from "!high" to "!hi"
    - Sends a confirmation message to the channel
    - No longer permanantly adds the role to the user
    - Removes the role after 60 seconds
    
##### Added a spam checker

> Developer comments: Sometimes messages are sent really quickly. This feature has some fun with that

    - If five messages are sent by the same user within ten seconds, they will receive a random response telling them to stop spamming

##### Made the *shut up* messages more abstract

> Developer comments: People don't just say shut up when they want someone to shut up, this tracks other types of shut up messages.

    - Every message with the words "shut" and "up" in it will receive a response
    - "stfu" will also now receive a response
    
##### Probability changes

> Developer comments: Some messages were happening too often, gotta nerf it.

    - Cute message probability in response to Eli changed from 5% to 3%, but increased the number of possible messages

---
### *January 4, 2023*
`v1.1.1 pre-release`
##### Gave users a role when a specific command is received

> Developer comments: We wanted to have a high people-only voice channel, so using this command would give people access to that voice channel.

    - Keyword "!high"
    - Permanantly adds the role to the user
---
### *December 26, 2022*
`v1.0.1 release`

##### Added functionality to react to messages from specific users

> Developer comments: This was the original inspiration for this bot. The aim was to troll some Discord users by reacting to every single one of their messages.

    - Thumbs up for Logan
    - X and thumbs down for Jocelyn
    - 5% chance to say something cute for Eli

##### Added a response to any *shut up* sent

> Developer comments: When people tell other people to shut up, wouldn't it be great if there were other messages agreeing with them?

    - Every "shut up" message will receive a response
    
#####  Added a a small probability to react to messages

> Developer comments: The amount of accuracte messages that this accidentally reacts to is uncanny.

    - 2% chance to react with F
    - 2% chance to react with a heart


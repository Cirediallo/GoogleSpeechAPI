# Report of Google Speech API

## French version

when using the french audio which is loud and clear the transcript of the audio not the translation shows some inperfection
just some examples:

> la SNCF assurera un train sur 3     
  
is transcripted as:  
  
> SNCF assurera un train sur 3    
    
where we notice that the french word **_la_** is not reported in the transcript    
   
> il ne senti ni douleur ni secousse    
   
is transcripted as:    
   
> ils sont inidhu leur nid secousses    
   
and we see clearly that the transcription is not at all related to the words in the audio    
   
> sois toujours plus têtu que la mer tu gagneras    
    
is transcripted as:    
   
> sois toujours à la mer tu gagneras    
same problem as above   
   
> Note: for the problem above double check with the team: 25-29s in the french audio   
    
> D'habitude j'ai des outils pour faire ça    
    
is trancripted as:    
    
> D'habitude j'ai dit vous qui pour faire ça    
      
> Nous nous sentions triste et très abattu    
   
is transctipted as:    
    
> nous nous sommes qu'ils ont triste et très abattu     
      
> Sa discrétion m'étonna énormement    
    
is transcripted as:    
      
> ce discrétion m'étonnait énormement    
      
Thing I noticed too is that the type of phrase is not reconized: is the sentence a question ? or exclamation !   
    
## English version   
     
The transcription of an english audio too have some problems   
     
> Note: double check with the team anyway     
the 6th transcription replaced **_fun_** by **_find_** and **_another word I didn't hear well_** by **_facebook_**   

for quotas and limitation check the web page: [Cloud Speech To Text API restrictions and limits](https://cloud.google.com/speech-to-text/quotas "Les limites et restrictions d'utilisation actuellement appliquées à l'API Cloud Speech to Text")    

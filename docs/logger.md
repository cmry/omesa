# Classes to log and display pipeline information


# Logger 

``` python 
 class Logger(file_name) 
```

Provides feedback to the user and can store settings in a log file.

Class holds a log string that can be formatted according to the used
components and is used to list settings that are provided to the
experiment. Aside from flat printing, it can iteratively store certain
lines that are reused (such as loading in multiple datasets). The save
function makes sure the self.log items are saved according to their logical
order.

| Parameters    | Type             | Doc             |
|:-------|:-----------------|:----------------|
        | fn | str | File name of the logfile (and the experiment). |
        

| Attributes    | Type             | Doc             |
|:-------|:-----------------|:----------------|
        | fn | str |         File name. |
        | buffer | list | Used to stack lines in a loop that can be written to the log line oncethe loop has been completed. |
        

--------- 

## Methods 

 

| Function    | Doc             |
|:-------|:----------------|
        | echo | Replacement for a print statement. Legacy function. |
        | loop | Print and store line to buffer. |
        | dump | Dump buffer to log. |
        | post | Print and store line to log. |
        | save | Save log. |
         
 

### echo

``` python 
    echo(*args) 
```


Replacement for a print statement. Legacy function.

### loop

``` python 
    loop(key, value) 
```


Print and store line to buffer.

### dump

``` python 
    dump(key) 
```


Dump buffer to log.

### post

``` python 
    post(key, value) 
```


Print and store line to log.

### save

``` python 
    save() 
```


Save log.
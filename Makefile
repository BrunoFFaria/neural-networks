# escolhe compilador
CC=gcc			
# flags a passar ao compilador 
# -O3 -mssse3 -align -xssse3 -axssse3
CFLAGS= -c  -Wall -g -O3 #-march=corei7 -xSSE4.2 -vec-report=2  -xHost -unroll -funroll-loops -use-intel-optimized-headers -fdata-sections   -falign-functions    -opt-mem-layout-trans=3  -ansi-alias -opt-mem-layout-trans=3
# flags de librarias
LDFLAGS =  -lm -L/usr/lib64 -llapack
# (ficheiros a compilar) para já fica assim, não sei o que o professor fernão quer fazer!!!
SOURCES = nntest.c neural_network.c

# extenções dos ficheiros seja objectos .o ou ficheiros de código 
OBJECTS=$(SOURCES:.c=.o)

# nome do ficheiro executavel desejado...
EXECUTABLE=nntest

# clean
RM = rm -f

# especifica um target, neste caso todos...
all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC)  $(OBJECTS) -o $@ $(LDFLAGS)

.c.o:
	$(CC)  $< -o $@ $(CFLAGS)

clean: 
	$(RM) *.o


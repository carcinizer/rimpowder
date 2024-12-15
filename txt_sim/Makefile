txt_sim_test: main.o txt_sim.o
	g++ main.o txt_sim.o -o txt_sim_test

main.o: main.cpp
	g++ -c main.cpp

txt_sim.o: txt_sim.cpp
	g++ -c txt_sim.cpp

clean:
	rm *.o txt_sim_test
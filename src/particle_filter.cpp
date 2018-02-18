/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>


#include "particle_filter.h"

using namespace std;


 

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	is_initialized = true;
	num_particles = 10;

	for(int i=0; i < num_particles; i++){
		weights.push_back(1.0);
	}
	
	random_device rd;
 default_random_engine gen(rd());
	
	// This line creates a normal (Gaussian) distribution for x
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	
	for(int i = 0; i< num_particles; i++){
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
        
        particles.push_back(p);

	}

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	random_device rd;
 default_random_engine gen(rd());
	// This line creates a normal (Gaussian) distribution for x
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
	//avoid division by zero
	for(int i=0; i<num_particles; i++){
		if (fabs(yaw_rate) > 0.0001) {
	        particles[i].x  +=  (velocity/yaw_rate) * ( sin (particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
	        particles[i].y  +=  (velocity/yaw_rate) * ( cos (particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
	        particles[i].theta += yaw_rate*delta_t;
	    }
	    else {
	       particles[i].x  += velocity* delta_t * cos(particles[i].theta);
	       particles[i].y  += velocity* delta_t * sin(particles[i].theta);
	    }

	    
		
		// adding noise
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
   

}

void ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> landMarks, std::vector<LandmarkObs>& transObservations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for(int i =0; i < transObservations.size();i++){
        
        double tx = transObservations[i].x;
        double ty = transObservations[i].y;

        double min = 999999.0;
        int id = 0;
  
    	for(int j = 0 ; j < landMarks.size();j++){
          
          double lx = landMarks[j].x_f;
          double ly = landMarks[j].y_f;
        
          if(dist(tx, ty, lx, ly) < min){
          	min = dist(tx, ty, lx, ly);
          	id = landMarks[j].id_i;
          }
    	}

    	transObservations[i].id = id;
    }
}

double ParticleFilter::gaussian_prob(float meanx, float meany, double x, double y, double std_landmark[]){

	double a = 1/(2 * M_PI * std_landmark[0] * std_landmark[1]);

	double x_denom = 2 * std_landmark[0] * std_landmark[0];
    double y_denom = 2 * std_landmark[1] * std_landmark[1];

    double x_diff = x - meanx;
    double y_diff = y - meany;

    double b = ((x_diff * x_diff)/x_denom) + ((y_diff * y_diff)/y_denom);

	return a * exp(-1 * b);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


	for(int i = 0; i < num_particles; i++){
		double xp = particles[i].x;
		double yp = particles[i].y;
		double thetap = particles[i].theta;

		std::vector<LandmarkObs> transformed;
		for(int j =0; j < observations.size(); j++){
			double xm = xp + (cos(thetap) * observations[j].x) - (sin(thetap) * observations[j].y);
			double ym = yp + (sin(thetap) * observations[j].x) + (cos(thetap) * observations[j].y);
			transformed.push_back({observations[j].id,xm,ym});
		}
        std::vector<Map::single_landmark_s> landMarks;
        for(int j =0; j < map_landmarks.landmark_list.size(); j++){
            float lmx = map_landmarks.landmark_list[j].x_f;
            float lmy = map_landmarks.landmark_list[j].y_f;

            if(fabs(xp-lmx) <= sensor_range && fabs(yp-lmy) <= sensor_range){
            	landMarks.push_back(map_landmarks.landmark_list[j]);
            }
        }
       
        dataAssociation(landMarks, transformed);

       // cout << "landMarks size new: " << landMarks.size() << endl ;
        //cout << "transformed size: " << transformed.size() << endl;
        
        particles[i].weight = 1;
        for(int j =0; j<transformed.size(); j++){
        	double tx = transformed[j].x;
        	double ty = transformed[j].y;

        	int matching_Landmark_id = transformed[j].id;

        	float lmx,lmy;
        	for(int k=0; k < landMarks.size(); k++){
        		if(landMarks[k].id_i == matching_Landmark_id){
        			lmx = landMarks[k].x_f;
        			lmy = landMarks[k].y_f;
        		}
        	}

            double w = gaussian_prob(lmx, lmy, tx, ty, std_landmark);
          
            	 particles[i].weight *= w;
         

        	//cout << "sub weight for particle " << i << ": " << particles[i].weight << endl;
        }
     
       weights[i] = particles[i].weight;
     // cout << "final w for particle " << i << ": " << weights[i] << endl;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Vector for new particles
	vector<Particle> new_particles (num_particles);
random_device rd;
 default_random_engine gen(rd());
	for (int i = 0; i < num_particles; ++i) {
	    discrete_distribution<int> index(weights.begin(),weights.end());
	    new_particles[i] = particles[index(gen)];
    }

    particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}


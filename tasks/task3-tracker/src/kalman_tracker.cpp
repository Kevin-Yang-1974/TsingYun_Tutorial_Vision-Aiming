#include "kalman_tracker.hpp"

#include <stdexcept>

namespace hw
{
    KalmanTracker::KalmanTracker() = default;

    bool KalmanTracker::isTracking() const
    {
        return tracking_;
    }

    void KalmanTracker::reset()
    {
        tracking_ = false;
        x_ = AxisFilter{};
        y_ = AxisFilter{};
        z_ = AxisFilter{};
    }

    void KalmanTracker::AxisFilter::reset(double measured_position)
    {
        position = measured_position;
        velocity = 0.0;
        p00 = 1.0;
        p01 = 0.0;
        p10 = 0.0;
        p11 = 1.0;
    }

    void KalmanTracker::AxisFilter::predict(double dt, double process_noise)
    {
        
        dt=std::max(dt, 0.0);
        position+=velocity*dt;
        double F00=1.0, F01=dt, F10=0.0, F11=1.0;
        double Q00=process_noise*dt*dt*dt*dt/4.0, Q01=process_noise*dt*dt*dt/2.0, Q10=process_noise*dt*dt*dt/2.0, Q11=process_noise*dt*dt;
        double new_p00=F00*p00*F00+F01*p10*F00+F00*p01*F01+F01*p11*F01+Q00;
        double new_p01=F00*p00*F10+F01*p10*F10+F00*p01*F11+F01*p11*F11+Q01;
        double new_p10=F10*p00*F00+F11*p10*F00+F10*p01*F01+F11*p11*F01+Q10;
        double new_p11=F10*p00*F10+F11*p10*F10+F10*p01*F11+F11*p11*F11+Q11;
        p00=new_p00;
        p01=new_p01;
        p10=new_p10;
        p11=new_p11;
    }

    void KalmanTracker::AxisFilter::update(double measured_position, double measurement_noise)
    {
        
        double residual=measured_position-position;
        double H0=1.0, H1=0.0;
        double S = H0 * p00 * H0 + H1 * p10 * H0 + H0 * p01 * H1 + H1 * p11 * H1 + measurement_noise;
        if(S<=0.0){
            return;
        }
        double K0 = (p00 * H0 + p01 * H1) / S;
        double K1 = (p10 * H0 + p11 * H1) / S;
        position += K0 * residual;
        velocity += K1 * residual;
        double new_p00 = (1.0 - K0 * H0) * p00 - K0 * H1 * p10;
        double new_p01 = (1.0 - K0 * H0) * p01 - K0 * H1 * p11;
        double new_p10 = (1.0 - K1 * H1) * p10 - K1 * H0 * p00;
        double new_p11 = (1.0 - K1 * H1) * p11 - K1 * H0 * p01;
        p00 = new_p00;
        p01 = new_p01;
        p10 = new_p10;
        p11 = new_p11;
    }
    KalmanTracker::AxisFilter KalmanTracker::AxisFilter::forecast(double dt, double process_noise) const
    {
        dt=std::max(dt, 0.0);
        double new_position=position+velocity*dt;
        double F00=1.0, F01=dt, F10=0.0, F11=1.0;
        double Q00=process_noise*dt*dt*dt*dt/4.0, Q01=process_noise*dt*dt*dt/2.0, Q10=process_noise*dt*dt*dt/2.0, Q11=process_noise*dt*dt;
        double new_p00=F00*p00*F00+F01*p10*F00+F00*p01*F01+F01*p11*F01+Q00;
        double new_p01=F00*p00*F10+F01*p10*F10+F00*p01*F11+F01*p11*F11+Q01;
        double new_p10=F10*p00*F00+F11*p10*F00+F10*p01*F01+F11*p11*F01+Q10;
        double new_p11=F10*p00*F10+F11*p10*F10+F10*p01*F11+F11*p11*F11+Q11;
        AxisFilter forecasted_filter;
        forecasted_filter.position=new_position;
        forecasted_filter.velocity=velocity;
        forecasted_filter.p00=new_p00;
        forecasted_filter.p01=new_p01;
        forecasted_filter.p10=new_p10;
        forecasted_filter.p11=new_p11;
        return forecasted_filter;
    }

    TrackState KalmanTracker::update(const Vec3 &measurement, double dt)
    {
        
        if(!tracking_){
            x_.reset(measurement.x);
            y_.reset(measurement.y);
            z_.reset(measurement.z);
            tracking_=true;
            return stateFromFilters();
        }
        x_.predict(dt, process_noise_);
        y_.predict(dt, process_noise_);
        z_.predict(dt, process_noise_);
        x_.update(measurement.x, measurement_noise_);
        y_.update(measurement.y, measurement_noise_);
        z_.update(measurement.z, measurement_noise_);
        return stateFromFilters();
    }

    TrackState KalmanTracker::predict(double dt)
    {
        
        if(!tracking_){
            return TrackState{};
        }
        AxisFilter x_predicted_ = x_.forecast(dt, process_noise_);
        AxisFilter y_predicted_ = y_.forecast(dt, process_noise_); 
        AxisFilter z_predicted_ = z_.forecast(dt, process_noise_);
        TrackState predicted_state;
        predicted_state.tracking=true;
        predicted_state.position={x_predicted_.position, y_predicted_.position, z_predicted_.position};
        predicted_state.velocity={x_predicted_.velocity, y_predicted_.velocity, z_predicted_.velocity};
        return predicted_state;
    }

    TrackState KalmanTracker::stateFromFilters() const
    {
        return {
            true,
            {x_.position, y_.position, z_.position},
            {x_.velocity, y_.velocity, z_.velocity},
        };
    }
} // namespace hw

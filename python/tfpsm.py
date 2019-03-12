import numpy as np
import numpy.random as rn
import tensorflow as tf
import tensorflow_probability as tfp
#import matplotlib.pyplot as plt
#%matplotlib inline
#plt.style.use('seaborn')
#import pickle

from absl import app,flags, logging

FLAGS = flags.FLAGS



nPVS = 160 #@param 
#@markdown Number of Subjects
nSub = 24 #@param 

#@markdown A standard deviation for typical PVS
std = 0.75 #@param 
#@markdown A parameter of the gamma distribution from which the variances are generated 
alpha = 30.0 #@param 
#@markdown Standard deviation for delta distribution
b = 0.3 #@param 

def make_prior():
    delta_prior = tfp.distributions.Normal(loc=np.float64(0.0),scale=np.float64(b) )
    psi_prior = tfp.distributions.Uniform(low=np.float64(1.0),
                                    high=np.float64(5.0) )
    #the same for fi
    upsilon_prior = tfp.distributions.Gamma(
        concentration=np.float64(alpha),
        rate=np.float64(alpha/np.sqrt(0.5*(std**2 - b**2))))

    fi_prior = tfp.distributions.Gamma(
        concentration=np.float64(alpha),
        rate=np.float64(alpha/np.sqrt(0.5*(std**2 - b**2))))
    return delta_prior,psi_prior,upsilon_prior,fi_prior

def make_prior_exp():
    delta_prior = tfp.distributions.Normal(loc=np.float64(0.0),scale=np.float64(0.3) )
    psi_prior = tfp.distributions.Uniform(low=np.float64(1.0),
                                          high=np.float64(5.0) )

    
    #the same for fi
    upsilon_prior = tfp.distributions.Exponential(rate=np.float64(10))

    fi_prior = tfp.distributions.Exponential(rate=np.float64(10))
    return delta_prior,psi_prior,upsilon_prior,fi_prior


def make_model(loc, scale):
    O = tfp.distributions.QuantizedDistribution(
        distribution=tfp.distributions.TransformedDistribution(
            distribution=tfp.distributions.Normal(loc=loc, scale=scale),
            bijector=tfp.bijectors.AffineScalar(shift=np.float64(-0.5))),
        low=np.float64(1.0),
        high=np.float64(5.0))
    
    #O = tfp.distributions.Normal(loc=loc,scale=scale)
    return O

def make_sample(nPVS,nSub):
  
    delta_prior,psi_prior,upsilon_prior,fi_prior = make_prior()
    
    delta = delta_prior.sample([1,nSub])
    psi = psi_prior.sample([nPVS,1])
    
    upsilon = upsilon_prior.sample((1,nSub))
    fi = fi_prior.sample((nPVS,1))
    
    loc = delta + psi 
    
    scale = tf.sqrt(upsilon**2 + fi**2 )

    O = make_model(loc, scale)
    return delta,psi,upsilon,fi,O.sample()

def log_joint(delta, psi, upsilon, fi,o,prior=True):
    delta_prior,psi_prior,upsilon_prior,fi_prior = make_prior()
    
    loc = delta + psi 
    scale = tf.sqrt(upsilon**2 + fi**2 )
    
    O = make_model(loc, scale)

    prior_log_prob = tf.reduce_sum(delta_prior.log_prob(delta)) + \
        tf.reduce_sum(psi_prior.log_prob(psi)) + \
        tf.reduce_sum(upsilon_prior.log_prob(upsilon)) + \
        tf.reduce_sum(fi_prior.copy().log_prob(fi)) 
    
    loglik = tf.reduce_sum(O.log_prob(o))
    ret = loglik + prior_log_prob if prior else loglik
    return ret

def make_plot(psi,psi_hat,delta,delta_hat,fi,fi_hat,upsilon,upsilon_hat,\
             psi_err=None,delta_err=None,fi_err = None, upsilon_err=None):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    
    fig,ax = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(10,10)

    ax[0,0].scatter(psi,psi_hat,zorder=3)
    ax[0,0].plot([1,5],[1,5],color=colors[1],zorder=2)
    ax[0,0].set_xlabel(r'$\psi$')
    ax[0,0].set_ylabel(r'$\hat \psi$')
    
    if psi_err is not None:
        ax[0,0].errorbar(x=psi,y=psi_hat, fmt='none',yerr=psi_err, alpha=0.9,ecolor=colors[2],zorder=1)

    ax[0,1].scatter(delta,delta_hat,zorder=3)
    ax[0,1].plot([-1,1.3],[-1,1.3],color=colors[1])
    ax[0,1].set_xlabel(r'$\delta$')
    ax[0,1].set_ylabel(r'$\hat \delta$')

    if delta_err is not None:
        ax[0,1].errorbar(x=delta,y=delta_hat, fmt='none',yerr=delta_err, alpha=0.9, ecolor=colors[2])
    
    ax[1,0].scatter(fi,fi_hat, zorder=3)
    ax[1,0].plot([0.2,0.8],[0.2,0.8],color=colors[1])
    ax[1,0].set_xlabel(r'$\phi$')
    ax[1,0].set_ylabel(r'$\hat \phi$')    

    if fi_err is not None:
        ax[1,0].errorbar(x=fi,y=fi_hat, fmt='none',yerr=fi_err, alpha=0.9,ecolor=colors[2],zorder=2)

    ax[1,1].scatter(upsilon,upsilon_hat, zorder=3)
    ax[1,1].plot([0.2,0.8],[0.2,0.8],color=colors[1])
    ax[1,1].set_xlabel(r'$\upsilon$')
    ax[1,1].set_ylabel(r'$\hat \upsilon$')

    if upsilon_err is not None:
        ax[1,1].errorbar(x=upsilon,y=upsilon_hat, fmt='none',yerr=upsilon_err, alpha=0.9,ecolor=colors[2],zorder=2)

def draw_sample(n):
    g = tf.Graph()
    with g.as_default():
        s = make_sample(n,n)
    with tf.Session(graph=g) as sess:
        delta,psi,upsilon,fi,o = sess.run(s)
    return delta,psi,upsilon,fi,o
   
def fit_mle(o):
    g = tf.Graph()
    with g.as_default():
        # random start conditions
        delta_s,psi_s,upsilon_s,fi_s,_ = make_sample(o.shape[0], o.shape[1])
        
        tf_delta = tf.Variable(delta_s)
        tf_psi_var = tf.Variable((psi_s-np.float64(1.0))/np.float64(4.0))
        tf_psi = np.float64(1.0) + np.float64(4.0) * tf.nn.sigmoid(tf_psi_var)

        
        tf_upsilon_sqr = tf.Variable(tf.sqrt(upsilon_s))
        tf_upsilon = tf_upsilon_sqr**2
        
        tf_fi_sqr = tf.Variable(tf.sqrt(fi_s))
        tf_fi = tf_fi_sqr**2
        
        log_prob = log_joint(tf_delta,tf_psi,tf_upsilon,tf_fi,o,prior=False)
        loss = -log_prob/(o.shape[0]* o.shape[1])
        train_op = tf.train.RMSPropOptimizer(learning_rate=0.02, epsilon=0.01).minimize(loss)
        
    with tf.Session(graph=g) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            _,np_loss = sess.run([train_op,loss])
            if i % 500 ==0:
                print(i,np_loss)

        (delta_hat, 
        psi_hat, 
        upsilon_hat, 
        fi_hat, 
        loss_val) = sess.run([tf_delta,tf_psi,tf_upsilon,tf_fi,loss])

    return delta_hat,  psi_hat, upsilon_hat, fi_hat



def fit_map(o):
    g = tf.Graph()
    with g.as_default():
        # random start conditions
        delta_s,psi_s,upsilon_s,fi_s,_ = make_sample(o.shape[0], o.shape[1])
        
        tf_delta = tf.Variable(delta_s)
        tf_psi_var = tf.Variable((psi_s-np.float64(1.0))/np.float64(4.0))
        tf_psi = np.float64(1.0) + np.float64(4.0) * tf.nn.sigmoid(tf_psi_var)

        
        tf_upsilon_sqr = tf.Variable(tf.log(upsilon_s))
        tf_upsilon = tf.exp(tf_upsilon_sqr)
        
        tf_fi_sqr = tf.Variable(tf.log(fi_s))
        tf_fi = tf.exp(tf_fi_sqr)
        
        log_prob = log_joint(tf_delta,tf_psi,tf_upsilon,tf_fi,o)
        loss = -log_prob/(o.shape[0]* o.shape[1])
        train_op = tf.train.RMSPropOptimizer(learning_rate=0.02, epsilon=0.01).minimize(loss)
        
    with tf.Session(graph=g) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            _,np_loss = sess.run([train_op,loss])
            if i % 500 ==0:
                print(i,np_loss)

        (delta_hat, 
        psi_hat, 
        upsilon_hat, 
        fi_hat, 
        loss_val) = sess.run([tf_delta,tf_psi,tf_upsilon,tf_fi,loss])

    return delta_hat,  psi_hat, upsilon_hat, fi_hat


def sigmoid(x):  
    return np.exp(-np.logaddexp(0, -x))



def fit_bayes(o):
    g = tf.Graph()
    with g.as_default():
        # random start conditions
        delta_s,psi_s,upsilon_s,fi_s,_ = make_sample(o.shape[0], o.shape[1])

        def target_log_prob_fn(delta_o, psi_o, upsilon_o, fi_o):
            """Unnormalized target density as a function of states."""
            psi = np.float64(1.0) + np.float64(4.0) * tf.nn.sigmoid(psi_o)
            upsilon = tf.exp(upsilon_o)
            fi = tf.exp(fi_o)
            return log_joint(delta_o, psi, upsilon, fi,o)
        
        num_results = 1000
        num_burnin_steps = 300
        
        states, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=[
                delta_s,
                (psi_s-np.float64(1.0))/np.float64(4.0),
                tf.log(upsilon_s),
                tf.log(fi_s)
            ],
            kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                step_size=0.01,
                num_leapfrog_steps=3))
        
    with tf.Session(graph=g) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        np_states,acepted = sess.run([states,kernel_results.is_accepted])

    np_states[1] = 1 + 4*sigmoid(np_states[1])

    np_states[2] = np.exp(np_states[2])
    np_states[3] = np.exp(np_states[3])

    (delta_hat, 
        psi_hat, 
        upsilon_hat, 
        fi_hat)=map(lambda x: np.mean(x,axis=0),np_states)

    return (delta_hat,  
    psi_hat, 
    upsilon_hat, 
    fi_hat,
    np.percentile(np_states[0],q=2.5,axis=0),
    np.percentile(np_states[0],q=97.5,axis=0),
    np.percentile(np_states[1],q=2.5,axis=0),
    np.percentile(np_states[1],q=97.5,axis=0),
    np.percentile(np_states[2],q=2.5,axis=0),
    np.percentile(np_states[2],q=97.5,axis=0),
    np.percentile(np_states[3],q=2.5,axis=0),
    np.percentile(np_states[3],q=97.5,axis=0))



def main(argv):
    del argv  # Unused.

    for rep in range(40):
        for n in np.linspace(20,200,10)
            delta,psi,upsilon,fi,o = draw_sample(n)
            mle_hat = fit_mle(o)
            map_hat = fit_map(o)
            bayes_hat = fit_bayes(o)

            fname = 'fit_{}_n_{}.npz'.format(rep,n)

            np.savez(fname,delta,psi,upsilon,fi,o, *mle_hat, *map_hat, *bayes_hat)

            logging.info(fname)





if __name__ == '__main__':
    app.run(main)
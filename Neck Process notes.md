This is a sort of unorganized aggregate of notes on the process so nothing gets lost.
## test woods
Quilt maple
Rock maple
Wenge (maybe)
Walnut
Mahogany
Sapelle
spanish cedar
Ash (maybe)
pine/spruce (extreme testing)
Aluminium/steel beam (other extreme)


## test considerations.
- suspend the whole setup, to allow maximum possible resonance/transmission
- clamp one end to a solid holder. more like a neck sticking out from a body
- add a soft object to nut end (the hand)
- strum the setup on one test, use this to normalize the transducer output relative to playing, then lock the transducer volume/gain
- use the transducer to create vibrations at the neck. Note: mount to neck sideways, as that is the direction of string vibration.
- another possibility is to use the 1k air core coil as a string exciter at the nut. This will more closely simulate an Ebow like effect. and does not need to be in contact with the wood, reducing mass errors, but does have spatial error problems.
- 


## spectral testing
- run frequency sweeps of the different woods. Try a couple different rates of sweep to see if it has an effect.
- compare the graph of the transducer to the graph of the measurements. Ideally a response of the pickup/micrphones will be available as well.
- pulse (impulse) test as well, approximating a tap, this should give an impulse response, might even get some formulas from this!
- select a handful of target frequencies to run steady state tests
- build all this out in a python script, automate as much as possible.


- 
## density vs species

## measuring
This is where things get tricky.  
At present the plan is to use a surface mount "surface exciter" transducer with frequency sweeps mounted at the nut position, and perhaps the bridge position to consistently excite the neck.
But to capture the sound, some ideas are:
- twin pencil microphones at opposite ends of the blank, comparing the two should infer frequency loss.
- Considerations, what are the responses of the Mics?  sensitive enough? what about crosstalk? (maybe a wall for this)
- Use a bridge plate transducer for acoustics.
- Benefits, this is guitar tech, so if it can't measure the effect, then no processed audio will show an effect. Also this can be used on any blank, no strings (haha) attached.
- downsides, what is the frequency response? Expense on purchase. Mounting to plate may introduce error.
- Mounting a pickup.
- Ok, this is straight up the electric guitar scenario, so this test must be done.
- Upsides, provides a direct measurement effect. If no sample shows change, then there is no tonewood.  I have the parts
- downsides. This is the most complex setup. Requires a bridge, nut, tuners, and pickup.  maintaining a consistent setup will be challenging.
- * to that end, create a jig that allows the parts to be attached consistently.


## error sources
- Transducer-string crosstalk,  mitigate by testing with the nut somehow isolated from the pickup, if transmission is measured, then the EM field of the transducer is directly affecting the strings. BAD
- neck blank size. Will need to cut one set of blanks to the same size. Ideally all samples should be the same size, but some pieces are too expensive to waste for testing.
- Neck blank density.  Need at least one species with multiple densities avaliable.  Maple is a prime candidate, use Hard and quilted, quilted is significantly less dense.  and I don't mind chopping some of this.
- Truss rod?  one blank with/without truss rod is a good idea. This will affect weight though.
- Bridge/nut mounting point tone absorbtion. Ideally these would be screwed into each blank, but i need to mount them on risers for the pickup to fit.
- weight of the rig at the nut end may dampen the entire test (ugh)  Try to keep the setup approximately the same weight as a reasonable headstock. (Gibson?)

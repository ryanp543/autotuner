#!/bin/bash
# Note that this is really just a quick fix. TODO: Replace with proper export when building the package ! 
cp ~/agrobot/build/robot_description/libset_trackfriction_plugin.so ~/agrobot/devel/lib
echo "move_to_devel.sh: The file set_trackfriction_plugin.so has been moved to the devel/lib folder."

exit 0
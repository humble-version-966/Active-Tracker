
# wall = [[-10,-15,0],[10,-15,0],[-10,15,0],[10,15,0],[-10,-15,10],[10,-15,10],[-10,15,10],[10,15,10]]
# wall_in_camera = Active_Tracker_Camera(wall)

# x_axis_people, y_axis_people = position_in_array(people_in_camera)
# x_axis_wall, y_axis_wall = position_in_array(wall_in_camera)

# for i in range(4):
#     Draw_cube(x_axis_people[i*8:i*8+8], y_axis_people[i*8:i*8+8])
#     Draw_cube(x_axis_wall[i*8:i*8+8], y_axis_wall[i*8:i*8+8])
#     Draw_box(x_axis_people[i*8:i*8+8], y_axis_people[i*8:i*8+8])

#     plt.xlabel("xvalue",fontsize = 10)
#     plt.ylabel("yvalue",fontsize = 10)
#     plt.xlim(0, 4256)
#     plt.ylim(0, 2832)

#     if i == 0:
#         plt.title('camera0')
#         plt.savefig("camera0")
#         plt.close()
#     if i == 1:
#         plt.title('camera1')
#         plt.savefig("camera1")
#         plt.close()
#     if i == 2:
#         plt.title('camera2')
#         plt.savefig("camera2")
#         plt.close()
#     if i == 3:
#         plt.title('camera3')
#         plt.savefig("camera3")
#         plt.close()


# door1 = [0,-15]
# door2 = [0,15]

# randomwalk2D(door1,door2,length,width)


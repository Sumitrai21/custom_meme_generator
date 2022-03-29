from face_swapping import FaceSwapper


face_swapper = FaceSwapper()
face_swapper.input_image("sumit.jpeg")
face_swapper.add_meme_template("sumit.jpeg")

face_swapper.create_swap()
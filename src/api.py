from flask import make_response, jsonify, request
from flask_restx import Api, Resource, fields, reqparse
from werkzeug.datastructures import FileStorage
import torch
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '.')
from src.model import ResnetModel
from src.preproc import preproc_image, classes

api = Api(
    version="1.0",
    title="Weather Classification API",
    description="API, предоставляющий интерфейс к модели определения погоды по снимку. Возможные классы: 'cloudy', 'snow', 'rain', 'fogsmog', 'shine'.",
    contact="adromanov_2@edu.hse.ru",
    default="APImethods",
    default_label="Функционал"
)

best_model = ResnetModel(n_classes=5)
with open('src/pickles/best_checkpoint.pth', 'rb') as f:
    checkpoint = torch.load(f, map_location=torch.device('cpu'))
best_model.load_state_dict(checkpoint)
best_model.eval()


model_true = api.model('predict_true_model', {
    "success_flg": fields.Boolean(description="Флаг успешного выполнения", example=True),
    "class": fields.String(description="Предсказанный класс", example="snow"),
})
model_false = api.model('predict_false_model', {
    "success_flg": fields.Boolean(description="Флаг успешного выполнения", example=False),
    "info": fields.String(description="Доп. информация о выполнении.", example="Bad image input.")
})
predicter_parser = reqparse.RequestParser()
predicter_parser.add_argument(
    'file',
    location='files',
    type=FileStorage,
    required=True,
    help="Файл с изображением")


@api.doc(description="Производит предсказание моделью на изображении, переданном в запросе.")
@api.expect(predicter_parser)
@api.response(200, 'Успешное выполнение', model_true)
@api.response(400, 'Неуспешное выполнение (доп. информация предоставляется в поле "info")', model_false)
class Predicter(Resource):
    def post(self):
        if 'file' not in request.files or request.files['file'].filename == '':
            result = {"success_flg": False, "info": "No file provided."}
        else:
            try:
                args = predicter_parser.parse_args()
                file = args["file"]
                img = preproc_image(file)
                prediction = best_model(img)[0].cpu().detach().numpy()
                prediction = classes[prediction.argmax()]
                result = {
                    "success_flg": True,
                    "class": prediction
                }
            except Exception:
                result = {
                    "success_flg": False,
                    "info": "Bad image input."
                }
        response = make_response(jsonify(result), int(
            200 * result["success_flg"] + 400 * (1 - result["success_flg"])))
        response.headers["Content-Type"] = "application/json"
        return response


api.add_resource(Predicter, "/predict")

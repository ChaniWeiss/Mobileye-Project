import part1
import part2
import part3
import matplotlib.pyplot as plt


class TflMan:
    def __init__(self):
        self.prev_res = []

    def run_on_frame(self, img_path, data_holder, model):
        fig, (light_src, tfl, distances) = plt.subplots(1, 3, figsize=(12, 6))
        res = {}
        res["light_sources"] = TflMan.find_light_src(img_path, light_src)

        param2 = {"img_path": img_path,
                  "candidates": res["light_sources"]["candidates"],
                  "auxiliary": res["light_sources"]["auxiliary"]}
        res["tfls"] = TflMan.verify_tfls(param2, tfl, model)
        assert len(res["light_sources"]["candidates"]) >= len(res["tfls"]["candidates"])

        if len(self.prev_res) == 0:
            self.prev_res = res["tfls"]["candidates"]
            plt.show()
            return res

        param3 = {"img_path": img_path,
                  "data_holder": data_holder,
                  "curr_candidates": res["tfls"]["candidates"]
                  }

        res["distances"] = self.find_distances(param3, distances)
        self.prev_res = res["tfls"]["candidates"]
        plt.show()
        return res

    @staticmethod
    def find_light_src(img_path, fig):
        return part1.find_lights(img_path, fig, "find_light_sources")


    @staticmethod
    def verify_tfls(param, fig, model):
        img_path = param["img_path"]
        candidades = param["candidates"]
        auxiliary = param["auxiliary"]
        return part2.verify_tfls(img_path, candidades, auxiliary, fig, "verify_tfls", model)

    def find_distances(self, param, fig):
        img_path = param["img_path"]
        data_holder = param["data_holder"]
        curr_candidates = param["curr_candidates"]
        return part3.calc_distances(img_path, data_holder, curr_candidates, self.prev_res, fig)

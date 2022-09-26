package main

import (
	"example/todo-go/mlcode"
	"io/ioutil"
	"net/http"

	"github.com/gin-gonic/gin"
)

type responseDescription struct {
	JobId       string `json:"jobId"	`
	Status      int    `json:"status"	`
	Name        string `json:"name"	`
	Region      string `json:"region"	`
	Description string `json:"description"	`

	CandidateId     int    `json:"candidateid"	`
	Position        int    `json:"position"	`
	Sex             int    `json:"sex"	`
	Citizenship     string `json:"citizenship"	`
	Age             int    `json:"age"	`
	Salary          int    `json:"salary"	`
	Langs           string `json:"langs"	`
	DriverLicense   string `json:"driverlicense"	`
	Subway          string `json:"subway"	`
	Skills          string `json:"skills"	`
	Employment      int    `json:"employment"	`
	Shedule         int    `json:"shedule"	`
	CandidateRegion string `json:"candidateregion"	`
}

func addTodo(context *gin.Context) {
	body := context.Request.Body
	x, _ := ioutil.ReadAll(body)
	// jsonData, err := json.Marshal(response)
	// if err != nil {
	// 	log.Fatal(err)
	// }
	result := mlcode.Execute(string(x))
	context.IndentedJSON(http.StatusCreated, result)
}

func main() {
	router := gin.Default() //our server
	// router.GET("/search", getTodos)

	router.POST("/search", addTodo)

	router.Run(":8080")

}
